type OutputPanelProps = Readonly<{
  phrase: string;
  correctedPhrase: string | null;
  backendStatus: string;
  currentPrediction: string;
  onAddPrediction: () => void;
  onRemoveLast: () => void;
  onClear: () => void;
  onCorrect: () => void;
  onSpeak: () => void;
}>;

export function OutputPanel({
  phrase,
  correctedPhrase,
  backendStatus,
  currentPrediction,
  onAddPrediction,
  onRemoveLast,
  onClear,
  onCorrect,
  onSpeak,
}: OutputPanelProps) {
  return (
    <aside className="flex h-full flex-col gap-5">
      <section className="rounded-[20px] border border-asl-border bg-asl-panel p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.3em] text-asl-muted">
          Estado do backend
        </p>
        <div className="mt-4 flex items-center justify-between rounded-2xl border border-asl-border bg-black/10 px-4 py-3">
          <span className="text-sm text-asl-text">API</span>
          <span className="text-sm font-medium text-asl-accent">
            {backendStatus}
          </span>
        </div>
      </section>

      <section className="rounded-[20px] border border-asl-border bg-asl-panel p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.3em] text-asl-muted">
          Composicao
        </p>
        <div className="mt-4 rounded-3xl border border-dashed border-[rgba(255,194,15,0.22)] bg-black/10 p-4">
          <p className="text-sm text-asl-muted">Token atual</p>
          <p className="mt-2 font-mono text-3xl text-asl-accent">
            {currentPrediction}
          </p>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onAddPrediction}
            className="rounded-full bg-asl-accent px-6 py-2.5 text-sm font-bold text-white dark:text-[#241800] shadow-sm transition-all active:scale-95 hover:brightness-110"
          >
            Adicionar
          </button>
          <button
            type="button"
            onClick={onRemoveLast}
            className="rounded-full border border-asl-border bg-asl-surface px-6 py-2.5 text-sm font-medium text-asl-text shadow-sm transition-all active:scale-95 hover:border-asl-accent"
          >
            Apagar último
          </button>
          <button
            type="button"
            onClick={onClear}
            className="rounded-full border border-asl-border bg-asl-surface px-6 py-2.5 text-sm font-medium text-asl-text shadow-sm transition-all active:scale-95 hover:border-asl-accent"
          >
            Limpar
          </button>
        </div>
      </section>

      <section className="flex flex-1 flex-col rounded-[20px] border border-asl-border bg-asl-panel p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.3em] text-asl-muted">
          Frase
        </p>
        <div className="mt-4 min-h-36 rounded-3xl border border-asl-border bg-black/10 p-4 text-lg leading-8 text-asl-text">
          {phrase ||
            "A frase aparece aqui conforme acumulas letras ou palavras."}
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onCorrect}
            className="rounded-full border border-asl-border bg-asl-surface px-6 py-2.5 text-sm font-medium text-asl-text transition-all hover:border-asl-accent hover:bg-asl-accent hover:text-black shadow-sm active:scale-95"
          >
            Corrigir frase
          </button>
          <button
            type="button"
            onClick={onSpeak}
            className="rounded-full border border-asl-border bg-asl-surface px-6 py-2.5 text-sm font-medium text-asl-text transition-all hover:border-asl-accent hover:bg-asl-accent hover:text-black shadow-sm active:scale-95"
          >
            Ler em voz alta
          </button>
        </div>
        <div className="mt-5 flex flex-1 flex-col rounded-3xl border border-asl-border bg-black/10 p-4">
          <p className="text-sm text-asl-muted">Versão corrigida</p>
          <p className="mt-2 text-base text-asl-text">
            {correctedPhrase ?? "Sem correção aplicada."}
          </p>
        </div>
      </section>
    </aside>
  );
}
