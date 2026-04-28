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
      <section className="rounded-[20px] border border-asl-border bg-[rgba(43,32,7,0.76)] p-5 backdrop-blur-xl">
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

      <section className="rounded-[20px] border border-asl-border bg-[rgba(43,32,7,0.76)] p-5 backdrop-blur-xl">
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
            className="rounded-full bg-asl-accent px-4 py-2 text-sm font-semibold text-[#241800]"
          >
            Adicionar
          </button>
          <button
            type="button"
            onClick={onRemoveLast}
            className="rounded-full border border-asl-border px-4 py-2 text-sm text-asl-text"
          >
            Apagar ultimo
          </button>
          <button
            type="button"
            onClick={onClear}
            className="rounded-full border border-asl-border px-4 py-2 text-sm text-asl-text"
          >
            Limpar
          </button>
        </div>
      </section>

      <section className="flex-1 rounded-[20px] border border-asl-border bg-[rgba(43,32,7,0.76)] p-5 backdrop-blur-xl">
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
            className="rounded-full bg-asl-surface px-4 py-2 text-sm font-medium text-asl-text"
          >
            Corrigir frase
          </button>
          <button
            type="button"
            onClick={onSpeak}
            className="rounded-full bg-asl-surface px-4 py-2 text-sm font-medium text-asl-text"
          >
            Ler em voz alta
          </button>
        </div>
        <div className="mt-5 rounded-3xl border border-asl-border bg-black/10 p-4">
          <p className="text-sm text-asl-muted">Versao corrigida</p>
          <p className="mt-2 text-base text-asl-text">
            {correctedPhrase ?? "Sem correcao aplicada."}
          </p>
        </div>
      </section>
    </aside>
  );
}
