type CameraPanelProps = Readonly<{
  videoRef: React.RefObject<HTMLVideoElement | null>;
  currentPrediction: string;
  confidence: number;
  mode: "static" | "dynamic";
  isRecording: boolean;
  onToggleMode: () => void;
  onToggleRecording: () => void;
}>;

export function CameraPanel(props: CameraPanelProps) {
  const {
    videoRef,
    currentPrediction,
    confidence,
    mode,
    isRecording,
    onToggleMode,
    onToggleRecording,
  } = props;

  return (
    <section className="relative overflow-hidden rounded-xl2 border border-asl-border bg-gradient-to-b from-[#080603] to-[#1a1204] shadow-panel">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="h-[520px] w-full scale-x-[-1] object-cover"
      />
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="max-w-xs rounded-3xl border border-[rgba(255,194,15,0.2)] bg-[rgba(24,17,3,0.75)] p-5 backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.3em] text-asl-muted">
              Predicao atual
            </p>
            <div className="mt-3 flex items-end gap-4">
              <span className="font-mono text-6xl font-semibold text-asl-accent">
                {currentPrediction}
              </span>
              <span className="mb-2 text-sm text-asl-muted">
                {Math.round(confidence * 100)}%
              </span>
            </div>
            <progress
              className="mt-4 h-2 w-full overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-white/10 [&::-webkit-progress-value]:bg-asl-accent [&::-moz-progress-bar]:bg-asl-accent"
              max={100}
              value={Math.round(confidence * 100)}
            />
          </div>
          <div className="rounded-full border border-asl-border bg-[rgba(24,17,3,0.75)] px-4 py-2 text-xs uppercase tracking-[0.25em] text-asl-muted backdrop-blur-xl">
            Webcam live
          </div>
        </div>
        <div className="pointer-events-auto flex items-center justify-center gap-3 self-center rounded-full border border-asl-border bg-[rgba(24,17,3,0.72)] p-3 backdrop-blur-xl">
          <button
            type="button"
            onClick={onToggleMode}
            className="rounded-full bg-asl-surface px-5 py-3 text-sm font-medium text-asl-text transition hover:bg-asl-accent hover:text-[#241800]"
          >
            Modo: {mode === "static" ? "Estático" : "Dinâmico"}
          </button>
          <button
            type="button"
            onClick={onToggleRecording}
            className={`rounded-full px-5 py-3 text-sm font-medium transition ${
              isRecording
                ? "bg-asl-warn text-white"
                : "bg-asl-surface text-asl-text hover:bg-asl-accent hover:text-[#241800]"
            }`}
          >
            {isRecording ? "Parar captura" : "Captura dinâmica"}
          </button>
        </div>
      </div>
    </section>
  );
}
