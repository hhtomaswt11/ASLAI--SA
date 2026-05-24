"use client";

import { useEffect, useRef, useState } from "react";

import { api, type HealthResponse } from "@/lib/api";
import { CameraPanel } from "@/components/camera-panel";
import { OutputPanel } from "@/components/output-panel";

type Mode = "static" | "dynamic";

/**
 * Ajustes de reconhecimento:
 * - Threshold mais alto: reduz falsos positivos.
 * - Mais frames de confirmação: obriga o gesto a ficar estável.
 * - Intervalo um pouco mais lento: continua fluido, mas menos nervoso.
 * - Cooldown depois de adicionar uma letra: evita adicionar demasiado depressa.
 */
const STATIC_CONFIDENCE_THRESHOLD = 0.62;
const STATIC_CONFIRM_HOLD_FRAMES = 7;
const STATIC_PREDICTION_INTERVAL_MS = 220;
const STATIC_COMMIT_COOLDOWN_MS = 1000;

function normalizeToken(token: string): string {
  return token.trim().toLowerCase();
}

function applyTokenToPhrase(current: string, token: string, mode: Mode): string {
  const normalized = normalizeToken(token);

  if (
    !normalized ||
    normalized === "--" ||
    normalized === "?" ||
    normalized === "nothing"
  ) {
    return current;
  }

  if (mode === "static") {
    if (normalized === "space") {
      return current && !current.endsWith(" ") ? `${current} ` : current;
    }

    if (normalized === "del") {
      return current.slice(0, -1);
    }

    return `${current}${token.toUpperCase()}`;
  }

  return current ? `${current} ${token}` : token;
}

export function TranslatorShell() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const dynamicFramesRef = useRef<string[]>([]);
  const staticRequestInFlightRef = useRef(false);
  const staticFailureCountRef = useRef(0);
  // Timer that auto-clears the prediction after a period of inactivity
  const predictionHoldTimerRef = useRef<number | null>(null);
  const PREDICTION_HOLD_MS = 4000; // keep prediction visible for 4 s

  const lastStaticTokenRef = useRef<string | null>(null);
  const consecutiveStaticHoldRef = useRef(0);
  const staticAutoAddedRef = useRef(false);
  const lastStaticCommitAtRef = useRef(0);

  const [mode, setMode] = useState<Mode>("static");
  const [isRecording, setIsRecording] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [backend, setBackend] = useState<HealthResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState("Offline");
  const [currentPrediction, setCurrentPrediction] = useState("--");
  const [confidence, setConfidence] = useState(0);
  const [phrase, setPhrase] = useState("");
  const [correctedPhrase, setCorrectPhrase] = useState<string | null>(null);
  const [recognizedToken, setRecognizedToken] = useState<string | null>(null);
  const [recognitionAnimationKey, setRecognitionAnimationKey] = useState(0);

  // Mantém o tema escuro/amarelo por defeito.
  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  useEffect(() => {
    let mounted = true;

    api
      .health()
      .then((payload) => {
        if (!mounted) return;
        setBackend(payload);
        setBackendStatus(`Online · MediaPipe ${payload.mediapipe_version}`);
      })
      .catch(() => {
        if (!mounted) return;
        setBackendStatus("Offline");
      });

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
        audio: false,
      })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(() => {
        setBackendStatus((previous) => `${previous} · Sem acesso a câmara`);
      });

    return () => {
      const tracks =
        (videoRef.current?.srcObject as MediaStream | null)?.getTracks() ?? [];
      tracks.forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    let isCancelled = false;

    if (mode !== "static") {
      if (captureTimerRef.current) {
        window.clearTimeout(captureTimerRef.current);
        captureTimerRef.current = null;
      }
      return;
    }

    const runStaticPrediction = async () => {
      if (isCancelled) return;

      if (staticRequestInFlightRef.current) {
        captureTimerRef.current = window.setTimeout(runStaticPrediction, 100);
        return;
      }

      const frame = captureFrame();
      if (!frame) {
        if (!isCancelled) {
          captureTimerRef.current = window.setTimeout(runStaticPrediction, 250);
        }
        return;
      }

      staticRequestInFlightRef.current = true;

      try {
        const result = await api.predictStatic(frame);
        if (isCancelled) return;

        const predictedToken = result.letter ?? "--";

        setCurrentPrediction(predictedToken);
        setConfidence(result.confidence);
        staticFailureCountRef.current = 0;

        handleStaticAutoCommit(
          result.letter,
          result.confidence,
          result.hand_detected,
        );
      } catch {
        if (isCancelled) return;
        setBackendStatus("Offline");
        staticFailureCountRef.current = Math.min(
          staticFailureCountRef.current + 1,
          5,
        );
      } finally {
        staticRequestInFlightRef.current = false;

        if (!isCancelled) {
          const delay =
            staticFailureCountRef.current > 0
              ? 1000 + staticFailureCountRef.current * 300
              : STATIC_PREDICTION_INTERVAL_MS;

          captureTimerRef.current = window.setTimeout(runStaticPrediction, delay);
        }
      }
    };

    captureTimerRef.current = window.setTimeout(runStaticPrediction, 0);

    return () => {
      isCancelled = true;
      if (captureTimerRef.current) {
        window.clearTimeout(captureTimerRef.current);
        captureTimerRef.current = null;
      }
      staticRequestInFlightRef.current = false;
    };
  }, [mode]);

  async function stopDynamicCaptureAndPredict() {
    setIsRecording(false);

    if (!dynamicFramesRef.current.length) return;

    try {
      const result = await api.predictDynamic(dynamicFramesRef.current);
      if (result.word) {
        // Got a confident result — show it and start the hold timer
        setCurrentPrediction(result.word);
        setConfidence(result.confidence);
        triggerRecognitionAnimation(result.word);

        // Reset the auto-clear timer so the prediction stays for PREDICTION_HOLD_MS
        if (predictionHoldTimerRef.current) {
          window.clearTimeout(predictionHoldTimerRef.current);
        }
        predictionHoldTimerRef.current = window.setTimeout(() => {
          setCurrentPrediction("--");
          setConfidence(0);
          predictionHoldTimerRef.current = null;
        }, PREDICTION_HOLD_MS);
      } else {
        // If result.word is null (low confidence / no hand), keep the previous
        // prediction visible — the hold timer will clear it when it expires.
        if (!predictionHoldTimerRef.current) {
          setCurrentPrediction("--");
          setConfidence(0);
        }
      }
    } finally {
      dynamicFramesRef.current = [];
    }
  }

  function captureFrame(): string | null {
    const video = videoRef.current;
    if (!video) return null;

    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    const context = canvas.getContext("2d");
    if (!context) return null;

    /**
     * Muito importante:
     * O vídeo está espelhado visualmente no CSS.
     * Aqui também espelhamos o frame enviado para o backend,
     * para ficar parecido com o cv2.flip(frame, 1) do notebook local.
     */
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.restore();

    return canvas.toDataURL("image/jpeg", 0.92).split(",")[1] ?? null;
  }

  function toggleTheme() {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }

  function toggleMode() {
    setMode((current) => (current === "static" ? "dynamic" : "static"));
    setCurrentPrediction("--");
    setConfidence(0);
    if (predictionHoldTimerRef.current) {
      window.clearTimeout(predictionHoldTimerRef.current);
      predictionHoldTimerRef.current = null;
    }
    resetStaticAutoCommit();
  }

  function toggleRecording() {
    if (mode !== "dynamic") {
      setMode("dynamic");
      return;
    }

    if (isRecording) {
      void stopDynamicCaptureAndPredict();
      return;
    }

    setIsRecording(true);
    dynamicFramesRef.current = [];
    // Clear any pending hold timer when a new recording starts
    if (predictionHoldTimerRef.current) {
      window.clearTimeout(predictionHoldTimerRef.current);
      predictionHoldTimerRef.current = null;
    }
    const interval = window.setInterval(() => {
      const frame = captureFrame();

      if (frame) {
        dynamicFramesRef.current.push(frame);
      }
      // Auto-stop at 60 frames — gives the backend close to the 64-frame
      // window the Transformer was trained on (captured at ~15 fps = 4 s)
      if (dynamicFramesRef.current.length >= 60) {
        window.clearInterval(interval);
        void stopDynamicCaptureAndPredict();
      }
    }, 67); // ~15 fps
  }

  function isValidRecognizedToken(token: string): boolean {
    const normalized = normalizeToken(token);

    return (
      Boolean(normalized) &&
      normalized !== "--" &&
      normalized !== "?" &&
      normalized !== "nothing"
    );
  }

  function formatRecognizedToken(token: string): string {
    const normalized = normalizeToken(token);

    if (normalized === "space") return "ESPAÇO";
    if (normalized === "del") return "APAGAR";

    return mode === "static" ? token.toUpperCase() : token;
  }

  function triggerRecognitionAnimation(token: string) {
    if (!isValidRecognizedToken(token)) return;

    setRecognizedToken(formatRecognizedToken(token));
    setRecognitionAnimationKey((current) => current + 1);
  }

  function resetStaticAutoCommit() {
    lastStaticTokenRef.current = null;
    consecutiveStaticHoldRef.current = 0;
    staticAutoAddedRef.current = false;
  }

  function commitTokenToPhrase(token: string, targetMode: Mode = mode) {
    if (!isValidRecognizedToken(token)) return;

    setPhrase((current) => applyTokenToPhrase(current, token, targetMode));
    setCorrectPhrase(null);
    triggerRecognitionAnimation(token);
  }

  function handleStaticAutoCommit(
    token: string | null | undefined,
    score: number,
    handDetected: boolean,
  ) {
    const normalized = normalizeToken(token ?? "");

    if (
      !handDetected ||
      !normalized ||
      normalized === "nothing" ||
      normalized === "--" ||
      normalized === "?" ||
      score < STATIC_CONFIDENCE_THRESHOLD
    ) {
      resetStaticAutoCommit();
      return;
    }

    if (normalized === lastStaticTokenRef.current) {
      consecutiveStaticHoldRef.current += 1;
    } else {
      lastStaticTokenRef.current = normalized;
      consecutiveStaticHoldRef.current = 1;
      staticAutoAddedRef.current = false;
    }

    const now = Date.now();
    const cooldownPassed =
      now - lastStaticCommitAtRef.current >= STATIC_COMMIT_COOLDOWN_MS;

    if (
      consecutiveStaticHoldRef.current >= STATIC_CONFIRM_HOLD_FRAMES &&
      !staticAutoAddedRef.current &&
      cooldownPassed
    ) {
      staticAutoAddedRef.current = true;
      lastStaticCommitAtRef.current = now;
      commitTokenToPhrase(token ?? "", "static");
    }
  }

  function addPredictionToPhrase() {
    const token = currentPrediction.trim();

    if (!token || token === "--") return;

    commitTokenToPhrase(token, mode);
  }

  function removeLast() {
    setPhrase((current) => {
      if (!current) return current;

      if (mode === "dynamic") {
        const parts = current.trim().split(/\s+/);
        parts.pop();
        return parts.join(" ");
      }

      return current.slice(0, -1);
    });
  }

  function clearPhrase() {
    setPhrase("");
    setCorrectPhrase(null);
    resetStaticAutoCommit();
  }

  async function correctPhrase() {
    if (!phrase) return;

    try {
      const response = await api.correctPhrase(phrase);
      setCorrectPhrase(response.corrected);
    } catch {
      setBackendStatus("Offline");
    }
  }

  async function speakPhrase() {
    const phraseText = (correctedPhrase ?? phrase).trim();
    const tokenText = currentPrediction.trim();
    const text =
      phraseText || (tokenText && tokenText !== "--" ? tokenText : "");

    if (!text) return;

    /**
     * Preferimos o TTS do browser porque toca diretamente no computador
     * do utilizador. O backend pode gerar MP3, mas pode não conseguir tocar
     * se o sistema não tiver mpv/ffplay/mplayer instalado.
     */
    if (
      typeof window !== "undefined" &&
      "speechSynthesis" in window &&
      "SpeechSynthesisUtterance" in window
    ) {
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 0.9;
      utterance.pitch = 1;

      window.speechSynthesis.speak(utterance);
      return;
    }

    try {
      await api.speak(text);
    } catch {
      setBackendStatus("Offline");
    }
  }

  return (
    <main className="relative mx-auto flex h-[100dvh] max-w-[1500px] flex-col overflow-hidden px-6 py-6 lg:px-10">
      <style>{`
        @keyframes asl-letter-jump {
          0% {
            transform: translateY(0) scale(1);
            box-shadow: 0 0 0 rgba(207, 185, 123, 0);
          }
          35% {
            transform: translateY(-14px) scale(1.14);
            box-shadow: 0 18px 35px rgba(207, 185, 123, 0.25);
          }
          70% {
            transform: translateY(2px) scale(0.98);
          }
          100% {
            transform: translateY(0) scale(1);
            box-shadow: 0 0 0 rgba(207, 185, 123, 0);
          }
        }

        .asl-letter-jump {
          animation: asl-letter-jump 520ms cubic-bezier(0.22, 1, 0.36, 1);
        }
      `}</style>

      <header className="mb-6 flex flex-col justify-between gap-6 rounded-[28px] border border-asl-border bg-asl-panel px-6 py-5 backdrop-blur-xl lg:flex-row lg:items-center">
        <div>
          <p className="text-sm uppercase tracking-[0.45em] text-[#cfb97b]">
            ASLAI
          </p>
          <h1 className="mt-3 text-3xl font-bold text-asl-text">
            Tradutor de Lingua Gestual Americana em tempo real
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={toggleTheme}
            className="flex h-10 w-10 items-center justify-center rounded-full border border-asl-border text-asl-text shadow-sm transition hover:bg-asl-accent hover:text-black"
            title="Alternar tema"
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>

          <div className="rounded-full border border-asl-border bg-asl-surface/30 px-4 py-2 text-sm text-asl-text shadow-sm">
            API: {backend ? "Ligada" : "A validar"}
          </div>

          <div className="rounded-full border border-asl-border bg-asl-surface/30 px-4 py-2 text-sm text-asl-text shadow-sm">
            Modelos:{" "}
            {backend?.static_model_loaded ? "Estático" : "Sem estático"} /{" "}
            {backend?.dynamic_model_loaded ? "Dinâmico" : "Sem dinâmico"}
          </div>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 gap-6 lg:grid-cols-[minmax(0,1fr),420px]">
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pb-4 pr-1">
          <CameraPanel
            videoRef={videoRef}
            currentPrediction={currentPrediction}
            confidence={confidence}
            mode={mode}
            isRecording={isRecording}
            onToggleMode={toggleMode}
            onToggleRecording={toggleRecording}
          />
          
          {recognizedToken && (
            <div
              aria-live="polite"
              className="flex flex-1 flex-col justify-center gap-2 rounded-[24px] border border-asl-border bg-asl-panel px-8 py-6 text-asl-text shadow-sm backdrop-blur-xl"
            >
              <p className="text-sm uppercase tracking-[0.3em] text-[#cfb97b]">
                Gesto reconhecido
              </p>

              <div className="mt-2 flex items-center gap-6">
                <span
                  key={recognitionAnimationKey}
                  className="asl-letter-jump flex h-24 min-w-[96px] items-center justify-center rounded-[20px] bg-asl-accent px-4 text-5xl font-black text-white dark:text-black shadow-lg"
                >
                  {recognizedToken}
                </span>

                <span className="text-base text-asl-muted whitespace-nowrap">
                  A animação aparece aqui quando o gesto é confirmado.
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pb-4 pr-1">
          <OutputPanel
            phrase={phrase}
            correctedPhrase={correctedPhrase}
            backendStatus={backendStatus}
            currentPrediction={currentPrediction}
            onAddPrediction={addPredictionToPhrase}
            onRemoveLast={removeLast}
            onClear={clearPhrase}
            onCorrect={correctPhrase}
            onSpeak={speakPhrase}
          />
        </div>
      </div>
    </main>
  );
}
