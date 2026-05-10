"use client";

import { useEffect, useRef, useState } from "react";

import { api, type HealthResponse } from "@/lib/api";
import { CameraPanel } from "@/components/camera-panel";
import { OutputPanel } from "@/components/output-panel";

type Mode = "static" | "dynamic";

function joinPhrase(current: string, token: string, mode: Mode): string {
  if (!token) return current;
  if (!current) return token;
  return mode === "dynamic" ? `${current} ${token}` : `${current}${token}`;
}

export function TranslatorShell() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const dynamicFramesRef = useRef<string[]>([]);
  const staticRequestInFlightRef = useRef(false);
  const staticFailureCountRef = useRef(0);

  const [mode, setMode] = useState<Mode>("static");
  const modeRef = useRef<Mode>(mode);
  const [isRecording, setIsRecording] = useState(false);
  const [backend, setBackend] = useState<HealthResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState("Offline");
  const [currentPrediction, setCurrentPrediction] = useState("--");
  const [confidence, setConfidence] = useState(0);
  const [phrase, setPhrase] = useState("");
  const [correctedPhrase, setCorrectedPhrase] = useState<string | null>(null);

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
    modeRef.current = mode;
    if (mode !== "static") {
      if (captureTimerRef.current) {
        window.clearTimeout(captureTimerRef.current);
        captureTimerRef.current = null;
      }
      return;
    }

    const runStaticPrediction = async () => {
      if (modeRef.current !== "static") return;

      if (staticRequestInFlightRef.current) {
        captureTimerRef.current = window.setTimeout(runStaticPrediction, 300);
        return;
      }

      const frame = captureFrame();
      if (!frame) {
        captureTimerRef.current = window.setTimeout(runStaticPrediction, 500);
        return;
      }

      staticRequestInFlightRef.current = true;

      try {
        const result = await api.predictStatic(frame);
        // Only apply static prediction if the UI is still in static mode
        if (modeRef.current === "static") {
          setCurrentPrediction(
            result.letter ?? (result.hand_detected ? "?" : "--"),
          );
          setConfidence(result.confidence);
        }
        staticFailureCountRef.current = 0;
      } catch {
        setBackendStatus("Offline");
        staticFailureCountRef.current = Math.min(
          staticFailureCountRef.current + 1,
          5,
        );
      } finally {
        staticRequestInFlightRef.current = false;

        const delay =
          staticFailureCountRef.current > 0
            ? 1800 + staticFailureCountRef.current * 400
            : 1200;
        captureTimerRef.current = window.setTimeout(runStaticPrediction, delay);
      }
    };

    captureTimerRef.current = window.setTimeout(runStaticPrediction, 0);

    return () => {
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
      setCurrentPrediction(result.word ?? "--");
      setConfidence(result.confidence);
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
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8).split(",")[1] ?? null;
  }

  function toggleMode() {
    setMode((current) => (current === "static" ? "dynamic" : "static"));
    setCurrentPrediction("--");
    setConfidence(0);
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
    const interval = window.setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        dynamicFramesRef.current.push(frame);
      }
      if (dynamicFramesRef.current.length >= 30) {
        window.clearInterval(interval);
        void stopDynamicCaptureAndPredict();
      }
    }, 120);
  }

  function addPredictionToPhrase() {
    const token = currentPrediction.trim();
    if (!token || token === "--") return;
    setPhrase((current) => joinPhrase(current, token, mode));
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
    setCorrectedPhrase(null);
  }

  async function correctPhrase() {
    if (!phrase) return;
    try {
      const response = await api.correctPhrase(phrase);
      setCorrectedPhrase(response.corrected);
    } catch {
      setBackendStatus("Offline");
    }
  }

  async function speakPhrase() {
    const phraseText = (correctedPhrase ?? phrase).trim();
    const tokenText = currentPrediction.trim();
    const text = phraseText || (tokenText && tokenText !== "--" ? tokenText : "");
    if (!text) return;
    try {
      await api.speak(text);
    } catch {
      setBackendStatus("Offline");
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-[1500px] flex-col px-6 py-8 lg:px-10">
      <header className="mb-6 flex flex-col justify-between gap-6 rounded-[28px] border border-asl-border bg-[rgba(24,17,3,0.82)] px-6 py-5 backdrop-blur-xl lg:flex-row lg:items-center">
        <div>
          <p className="text-sm uppercase tracking-[0.45em] text-asl-muted">
            ASLAI
          </p>
          <h1 className="mt-3 text-3xl font-bold text-asl-text">
            Tradutor de Lingua Gestual Americana em tempo real
          </h1>
        </div>
        <div className="flex flex-wrap gap-3">
          <div className="rounded-full border border-asl-border px-4 py-2 text-sm text-asl-text">
            API: {backend ? "Ligada" : "A validar"}
          </div>
          <div className="rounded-full border border-asl-border px-4 py-2 text-sm text-asl-text">
            Modelos:{" "}
            {backend?.static_model_loaded ? "Estático" : "Sem estático"} /{" "}
            {backend?.dynamic_model_loaded ? "Dinâmico" : "Sem dinâmico"}
          </div>
        </div>
      </header>

      <div className="grid flex-1 gap-6 lg:grid-cols-[minmax(0,1fr),420px]">
        <CameraPanel
          videoRef={videoRef}
          currentPrediction={currentPrediction}
          confidence={confidence}
          mode={mode}
          isRecording={isRecording}
          onToggleMode={toggleMode}
          onToggleRecording={toggleRecording}
        />
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
    </main>
  );
}
