const { useEffect, useState } = React;

function App() {
  const [documents, setDocuments] = useState([]);
  const [file, setFile] = useState(null);
  const [crop, setCrop] = useState("");
  const [year, setYear] = useState(new Date().getFullYear());
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadDocuments = () => fetch("/api/documents").then(r => r.json()).then(data => setDocuments(data.documents || []));
  useEffect(() => { loadDocuments(); }, []);

  async function upload(event) {
    event.preventDefault();
    if (!file || !crop.trim()) return setError("Choose a PDF and enter its crop.");
    setBusy(true); setError("");
    const form = new FormData();
    form.append("file", file); form.append("crop", crop); form.append("publication_year", year);
    try {
      const response = await fetch("/api/documents", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");
      setFile(null); event.target.reset(); await loadDocuments();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    const text = question.trim(); setQuestion(""); setBusy(true); setError("");
    setMessages(current => [...current, { role: "user", text }]);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Chat request failed");
      setMessages(current => [...current, { role: "assistant", text: data.answer, sources: data.sources }]);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  return <main>
    <header><div className="eyebrow">AGRICULTURAL KNOWLEDGE BASE</div><h1>Crop Care<span>.</span></h1><p>Grounded answers from your field library.</p></header>
    <section className="workspace">
      <aside className="library panel">
        <div className="panel-heading"><h2>Document library</h2><span>{documents.length} files</span></div>
        <form onSubmit={upload} className="upload-form">
          <label className="dropzone"><input type="file" accept="application/pdf" onChange={e => setFile(e.target.files[0])} /><strong>{file ? file.name : "Drop a PDF here"}</strong><small>PDF only, up to your server limit</small></label>
          <div className="fields"><label>Crop<input value={crop} onChange={e => setCrop(e.target.value)} placeholder="e.g. wheat" /></label><label>Year<input type="number" value={year} onChange={e => setYear(e.target.value)} min="1000" max={new Date().getFullYear()} /></label></div>
          <button disabled={busy} type="submit">{busy ? "Processing..." : "Add to library"}</button>
        </form>
        <div className="documents">{documents.length === 0 ? <p className="muted">No documents yet. Add a field guide to begin.</p> : documents.map(doc => <article className="document" key={doc.filename}><div className="pdf-mark">PDF</div><div><strong>{doc.filename}</strong><small>{doc.crop} · {doc.publication_year} · {doc.chunks} chunks</small></div><i>READY</i></article>)}</div>
      </aside>
      <section className="chat panel"><div className="panel-heading"><div><h2>Ask Crop Care</h2><span className="online"><b></b> Retrieval online</span></div><span className="model">GROUNDED MODE</span></div><div className="conversation">{messages.length === 0 && <div className="welcome"><div className="sprout">✦</div><h3>What are you working on?</h3><p>Ask about disease symptoms, crop practices, or field management. Answers stay anchored to your uploaded sources.</p><div className="suggestions"><button onClick={() => setQuestion("How can I manage wheat rust?")}>Manage wheat rust</button><button onClick={() => setQuestion("What are good rice practices?")}>Rice practices</button></div></div>}{messages.map((message, i) => <div className={`message ${message.role}`} key={i}><div className="bubble">{message.text}</div>{message.sources && <details><summary>{message.sources.length} source{message.sources.length === 1 ? "" : "s"} used</summary>{message.sources.map(source => <div className="source" key={source.chunk_id}><strong>{source.metadata.source}</strong><span>p. {source.metadata.page_start || source.metadata.page} · {(source.similarity * 100).toFixed(0)}% match</span><p>{source.text}</p></div>)}</details>}</div>)}</div><form className="chat-form" onSubmit={ask}><input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your crops..." maxLength="1000" /><button disabled={busy || !question.trim()} aria-label="Send">→</button></form></section>
    </section>{error && <div className="error">{error}</div>}<footer>Crop Care / Evidence-led agricultural support</footer>
  </main>;
}
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
