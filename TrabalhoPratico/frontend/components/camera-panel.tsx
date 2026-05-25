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
    <section className="relative flex flex-col rounded-xl2 border border-asl-border bg-asl-surface p-4">
      {/* Container do Video */}
      <div className="relative overflow-hidden rounded-[20px] bg-black/10">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="h-[520px] w-full scale-x-[-1] object-cover"
        />
        
        {/* Overlay superior (Predição) - Mantém-se sobre o vídeo */}
        <div className="pointer-events-none absolute inset-0 p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-xs rounded-3xl border border-asl-border bg-asl-panel p-5 backdrop-blur-xl">
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
                className="mt-4 h-2 w-full overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-asl-border [&::-webkit-progress-value]:bg-asl-accent [&::-moz-progress-bar]:bg-asl-accent"
                max={100}
                value={Math.round(confidence * 100)}
              />
            </div>
            <div className="rounded-full border border-asl-border bg-asl-panel px-4 py-2 text-xs uppercase tracking-[0.25em] text-asl-muted backdrop-blur-xl">
              Webcam live
            </div>
          </div>
        </div>
      </div>

      {/* Controlos - Agora abaixo do vídeo */}
      <div className="mt-4 flex items-center justify-center">
        <div className="pointer-events-auto flex items-center justify-center gap-4 rounded-full border border-asl-border bg-asl-panel p-3 shadow-sm backdrop-blur-xl">
          <button
            type="button"
            onClick={onToggleMode}
            className="rounded-full bg-asl-surface px-6 py-2.5 text-sm font-medium text-asl-text shadow-sm transition hover:bg-asl-accent hover:text-[#241800]"
          >
            Modo: {mode === "static" ? "Estático" : "Dinâmico"}
          </button>
          
          {mode === "dynamic" && (
            <div className="flex items-center gap-3 pr-2">
              <button
                type="button"
                onClick={onToggleRecording}
                className={`group flex h-12 w-12 items-center justify-center rounded-full border-2 transition-all duration-300 ${
                  isRecording 
                    ? "border-red-500 bg-red-500/10 shadow-[0_0_15px_rgba(239,68,68,0.3)]" 
                    : "border-asl-border bg-asl-surface hover:border-asl-accent"
                }`}
                title={isRecording ? "Parar captura" : "Iniciar captura"}
              >
                <div className={`transition-all duration-300 ${
                  isRecording 
                    ? "h-4 w-4 rounded-[2px] bg-red-500" 
                    : "h-5 w-5 rounded-full bg-red-600 group-hover:scale-110"
                }`} />
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
