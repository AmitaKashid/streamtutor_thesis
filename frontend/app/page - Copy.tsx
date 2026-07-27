"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Clock3,
  Cpu,
  Mic,
  RotateCcw,
  Search,
  Send,
  Settings2,
  Square,
  Volume2,
} from "lucide-react";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AnimatePresence, motion } from "framer-motion";


type Stage =
  | "idle"
  | "listening"
  | "transcribing"
  | "retrieving"
  | "thinking"
  | "streaming"
  | "speaking"
  | "interrupted"
  | "error";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  text: string;
  streaming?: boolean;
  timestamp: string;
}

interface SourceItem {
  id: string;
  title: string;
  subtitle?: string;
  snippet?: string;
  score?: number;
}

interface MetricState {
  stt?: number;
  ragLatency?: number;
  ttft?: number;
  ttsFirstChunk?: number;
  llmTotal?: number;
  ttsTotal?: number;
  firstAudio?: number;
  total?: number;
  promptChars?: number;
  promptWords?: number;
  conversationTurns?: number;
  conversationChars?: number;
}

interface TurnMetricRecord {
  turn: number;
  metrics: MetricState;
}



interface TutorConfig {
  sttBackend: string;
  llmModel: string;
  ttsBackend: string;
  qwen3Speaker: string;
  qwen3Language: string;
  ragEnabled: boolean;
  ragTopK: number;
  teachingStyle: string;
  conversationContext: boolean;
  conversationMaxTurns: string;
}

type ServerEvent =
  | { type: "stage"; value: Stage }
  | { type: "llm_chunk"; text: string }
  | { type: "assistant_done" }
  | { type: "sources"; data: SourceItem[] }
  | { type: "metrics"; data: MetricState }
  | { type: "captured_text"; text: string }
  | { type: "transcript_partial"; text: string }
  | { type: "transcript_final"; text: string }
  | { type: "user_transcript_final"; text: string }
  | { type: "barge_in_accepted"; text: string }
  | { type: "audio_chunk"; data: string; sample_rate: number; channels: number }
  | { type: "audio_done" }
  | { type: "audio_stop" }
  | {
    type: "config_applied";
    data: {
      llm_model?: string;
      llm_backend?: string;
      llm_server_url?: string;
      tts_backend?: string;
      rag?: boolean;
      rag_top_k?: number;
      teaching_style?: string;
      conversation_context?: boolean;
      conversation_max_turns?: number;
    };
  }
  | { type: "error"; message: string };

const LLM_MODELS = [
  { value: "llama3_8b_q4", label: "Llama 3 8B Q4" },
  { value: "gemma4_26b_q3", label: "Gemma 4 26B Q3" },
  { value: "qwen3_5_9b_q6", label: "Qwen 3.5 9B Q6" },
];

const TTS_BACKENDS = [
  { value: "kokoro", label: "Kokoro" },
  { value: "chatterbox", label: "Chatterbox" },
  { value: "chatterbox_stream", label: "Chatterbox Stream" },
  { value: "voxtral", label: "Voxtral" },
  { value: "qwen3", label: "Qwen3 TTS" },
];

const STT_BACKENDS = [
  { value: "faster-whisper", label: "Faster-Whisper" },
  { value: "mega-asr", label: "Mega-ASR" },
  { value: "parakeet", label: "Parakeet-TDT 0.6B v2" },
];

const QWEN3_SPEAKERS = [
  { value: "Vivian", label: "Vivian — Warm, conversational" },
  { value: "Serena", label: "Serena — Elegant, calm" },
  { value: "Ryan", label: "Ryan" },
  { value: "Aiden", label: "Aiden" },
  { value: "Ono_Anna", label: "Ono Anna" },
  { value: "Sohee", label: "Sohee" },
  { value: "Uncle_Fu", label: "Uncle Fu" },
  { value: "Dylan", label: "Dylan" },
  { value: "Eric", label: "Eric — Expressive" },
];

const QWEN3_LANGUAGES = [
  { value: "en", label: "English" },
  { value: "zh", label: "Chinese" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "de", label: "German" },
  { value: "fr", label: "French" },
  { value: "ru", label: "Russian" },
  { value: "pt", label: "Portuguese" },
  { value: "es", label: "Spanish" },
  { value: "it", label: "Italian" },
];

const TEACHING_STYLES = [
  { value: "socratic", label: "Socratic" },
  { value: "adaptive", label: "Adaptive" },
  { value: "inquiry", label: "Inquiry" },
  { value: "collaborative", label: "Collaborative" },
];

const stageLabel: Record<Stage, string> = {
  idle: "Ready",
  listening: "Listening",
  transcribing: "Transcribing",
  retrieving: "Retrieving",
  thinking: "Thinking",
  streaming: "Streaming",
  speaking: "Speaking",
  interrupted: "Interrupted",
  error: "Error",
};

const stageOrder: Stage[] = [
  "listening",
  "transcribing",
  "retrieving",
  "thinking",
  "streaming",
  "speaking",
];

const defaultConfig: TutorConfig = {
  sttBackend: "faster-whisper",
  llmModel: "llama3_8b_q4",
  ttsBackend: "kokoro",
  qwen3Speaker: "Vivian",
  qwen3Language: "en",
  ragEnabled: true,
  ragTopK: 2,
  teachingStyle: "socratic",
  conversationContext: true,
  conversationMaxTurns: "5",
};

function classNames(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function createId(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

const normalizeSttText = (text: string) =>
  (text || "")
    .toLowerCase()
    .replace(/\[[^\]]*\]/g, " ")
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^a-z0-9\s'-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const isSttHallucinationText = (text: string) => {
  const norm = normalizeSttText(text);

  if (!norm) return true;

  const exact = new Set([
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "bye bye",
    "bye-bye",
    "goodbye",
    "see you next time",
    "that's all",
  ]);

  if (exact.has(norm)) return true;

  const substrings = [
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "subscribe to my channel",
    "don't forget to subscribe",
    "please subscribe",
    "hit the bell",
    "turn on notifications",
    "see you in the next video",
    "subtitles by",
    "captioned by",
    "transcribed by",
    "music playing",
    "applause",
  ];

  return substrings.some((phrase) => norm.includes(phrase));
};

function formatSeconds(value?: number) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return `${value.toFixed(3)} s`;
}

function toBackendConfig(config: TutorConfig) {
  return {
    stt_backend: config.sttBackend,
    llm_model: config.llmModel,
    tts_backend: config.ttsBackend,
    qwen3_speaker: config.qwen3Speaker,
    qwen3_language: config.qwen3Language,
    rag: config.ragEnabled,
    rag_top_k: config.ragTopK,
    teaching_style: config.teachingStyle,
    conversation_context: config.conversationContext,
    conversation_max_turns:
      config.conversationContext && config.conversationMaxTurns.trim()
        ? Number(config.conversationMaxTurns)
        : 0,
  };
}

function useTutorSocket(url: string) {
  const [connected, setConnected] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [metrics, setMetrics] = useState<MetricState>({});
  const [metricHistory, setMetricHistory] = useState<TurnMetricRecord[]>([]);
  const [capturedText, setCapturedText] = useState("");
  const [partialText, setPartialText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const currentTurnMetricsRef = useRef<MetricState>({});
  const hasRecordedCurrentTurnRef = useRef(false);


  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackQueueRef = useRef<AudioBuffer[]>([]);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const browserAudioPlayingRef = useRef(false);

  const getPlaybackContext = () => {
    if (!playbackContextRef.current) {
      playbackContextRef.current = new AudioContext();
    }

    return playbackContextRef.current;
  };

  const decodePcm16Base64 = (
    base64: string,
    sampleRate: number,
    channels: number,
  ): AudioBuffer => {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    const pcm16 = new Int16Array(bytes.buffer);
    const frameCount = Math.floor(pcm16.length / channels);

    const ctx = getPlaybackContext();
    const buffer = ctx.createBuffer(channels, frameCount, sampleRate);

    for (let ch = 0; ch < channels; ch++) {
      const channelData = buffer.getChannelData(ch);

      for (let i = 0; i < frameCount; i++) {
        channelData[i] = pcm16[i * channels + ch] / 32768;
      }
    }

    return buffer;
  };

  const playNextAudioBuffer = () => {
    if (currentSourceRef.current || playbackQueueRef.current.length === 0) {
      return;
    }

    const ctx = getPlaybackContext();

    if (ctx.state === "suspended") {
      void ctx.resume();
    }

    const buffer = playbackQueueRef.current.shift();
    if (!buffer) return;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    currentSourceRef.current = source;
    browserAudioPlayingRef.current = true;

    source.onended = () => {
      currentSourceRef.current = null;

      if (playbackQueueRef.current.length > 0) {
        playNextAudioBuffer();
      } else {
        browserAudioPlayingRef.current = false;

        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "browser_audio_finished" }));
        }
      }
    };

    source.start();
  };

  const stopBrowserAudio = () => {
    playbackQueueRef.current = [];

    try {
      currentSourceRef.current?.stop();
    } catch {
      // ignore browser audio stop errors
    }

    currentSourceRef.current = null;
    browserAudioPlayingRef.current = false;

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "browser_audio_finished" }));
    }
  };



  useEffect(() => {
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (disposed) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed || wsRef.current !== ws) return;
        setConnected(true);
        setError(null);
      };

      ws.onclose = () => {
        if (disposed || wsRef.current !== ws) return;
        setConnected(false);
        reconnectTimer = setTimeout(connect, 1000);
      };

      ws.onerror = () => {
        if (disposed || wsRef.current !== ws) return;
        setConnected(false);
        setStage("error");
        setError("Backend websocket is not reachable.");
        try {
          ws.close();
        } catch {
          // ignore websocket close errors
        }
      };

      ws.onmessage = (event) => {
        try {
          const data: ServerEvent = JSON.parse(event.data);

          if (data.type === "config_applied") {
            console.log("Backend applied config:", data.data);
            return;
          }
          if (data.type === "audio_chunk") {
            const buffer = decodePcm16Base64(
              data.data,
              data.sample_rate,
              data.channels || 1,
            );

            playbackQueueRef.current.push(buffer);
            playNextAudioBuffer();
            return;
          }

          if (data.type === "audio_stop") {
            stopBrowserAudio();
            return;
          }

          if (data.type === "audio_done") {
            if (playbackQueueRef.current.length === 0 && !currentSourceRef.current) {
              browserAudioPlayingRef.current = false;
            }
            return;
          }
          if (data.type === "stage") {
            setStage(data.value);
            return;
          }

          if (data.type === "transcript_partial") {
            const text = data.text || "";

            if (isSttHallucinationText(text)) {
              setPartialText("");
              return;
            }

            setPartialText(text);
            setStage("transcribing");
            return;
          }

          if (data.type === "transcript_final" || data.type === "captured_text") {
            const text = (data.text || "").trim();

            if (isSttHallucinationText(text)) {
              setCapturedText("");
              setPartialText("");
              return;
            }

            setCapturedText(text);
            setPartialText("");
            return;
          }

          if (data.type === "user_transcript_final") {
            const text = (data.text || "").trim();
            if (isSttHallucinationText(text)) {
              setCapturedText("");
              setPartialText("");
              return;
            }
            if (text) {
              currentTurnMetricsRef.current = {};
              hasRecordedCurrentTurnRef.current = false;
              setMetrics({});

              setMessages((prev) => [
                ...prev,
                {
                  id: createId("user_audio"),
                  role: "user",
                  text,
                  timestamp: new Date().toLocaleTimeString(),
                },
              ]);
            }

            setCapturedText("");
            setPartialText("");
            return;
          }

          if (data.type === "barge_in_accepted") {
            const text = (data.text || "").trim();
            if (isSttHallucinationText(text)) {
              setCapturedText("");
              setPartialText("");
              return;
            }
            if (text) {
              currentTurnMetricsRef.current = {};
              hasRecordedCurrentTurnRef.current = false;
              setMetrics({});

              setMessages((prev) => [
                ...prev,
                {
                  id: createId("user_barge_in"),
                  role: "user",
                  text,
                  timestamp: new Date().toLocaleTimeString(),
                },
              ]);
            }

            setCapturedText("");
            setPartialText("");
            setStage("interrupted");
            return;
          }

          if (data.type === "llm_chunk") {
            setStage("streaming");
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant" && last.streaming) {
                next[next.length - 1] = { ...last, text: last.text + data.text };
                return next;
              }
              next.push({
                id: createId("assistant"),
                role: "assistant",
                text: data.text,
                streaming: true,
                timestamp: new Date().toLocaleTimeString(),
              });
              return next;
            });
            return;
          }

          if (data.type === "assistant_done") {
            setMessages((prev) =>
              prev.map((m, idx) =>
                idx === prev.length - 1 && m.role === "assistant"
                  ? { ...m, streaming: false }
                  : m,
              ),
            );

            const completedMetrics: MetricState = {
              ...currentTurnMetricsRef.current,
            };

            const hasAnyMetric = Object.values(completedMetrics).some(
              (value) => value !== undefined && value !== null,
            );

            if (hasAnyMetric && !hasRecordedCurrentTurnRef.current) {
              hasRecordedCurrentTurnRef.current = true;

              setMetricHistory((hist) => [
                ...hist,
                {
                  turn: hist.length + 1,
                  metrics: completedMetrics,
                },
              ]);
            }

            setStage("idle");
            return;
          }

          if (data.type === "sources") {
            setSources(data.data || []);
            return;
          }

          if (data.type === "metrics") {
            const incoming = data.data;

            setMetrics((prev) => {
              const merged: MetricState = {
                ...prev,
                ...incoming,
                stt:
                  incoming.stt !== undefined && incoming.stt > 0
                    ? incoming.stt
                    : prev.stt,
              };

              currentTurnMetricsRef.current = {
                ...currentTurnMetricsRef.current,
                ...incoming,
                stt:
                  incoming.stt !== undefined && incoming.stt > 0
                    ? incoming.stt
                    : currentTurnMetricsRef.current.stt ?? prev.stt,
              };


              return merged;
            });

            return;
          }

          if (data.type === "error") {
            setError(data.message);
            setStage("error");
          }
        } catch {
          setError("Received malformed websocket event.");
          setStage("error");
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      const ws = wsRef.current;
      wsRef.current = null;
      try {
        ws?.close();
      } catch {
        // ignore websocket close errors
      }
    };
  }, [url]);

  const send = (payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(payload));
    return true;
  };

  const resetTurnMetrics = () => {
    currentTurnMetricsRef.current = {};
    hasRecordedCurrentTurnRef.current = false;
  };

  return {
    connected,
    stage,
    messages,
    sources,
    metrics,
    metricHistory,
    setMetrics,
    capturedText,
    partialText,
    error,
    send,
    resetTurnMetrics,
    setStage,
    setMessages,
    setCapturedText,
    setPartialText,
    setError,
  };
}



function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="text-[11px] font-bold uppercase tracking-wide text-slate-500">{children}</label>;
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <div className="h-[68px] rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
      <div className="mb-2 text-[11px] font-bold uppercase tracking-wide text-slate-500">
        {label}
      </div>

      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-6 w-full appearance-none bg-transparent pr-7 text-sm font-medium text-slate-900 outline-none"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <ChevronDown className="pointer-events-none absolute right-0 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
      </div>
    </div>
  );
}
function NumberField({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const numericValue = Number(value || 0);

  const updateValue = (next: number) => {
    onChange(String(Math.max(0, next)));
  };

  return (
    <div className="h-[68px] rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
      <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-slate-500">
        {label}
      </div>

      <div className="flex items-center justify-between gap-2">
        <input
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          disabled={disabled}
          value={value}
          onChange={(e) => {
            const onlyNumbers = e.target.value.replace(/\D/g, "");
            onChange(onlyNumbers);
          }}
          className="h-7 w-full bg-transparent text-sm font-medium text-slate-900 outline-none disabled:text-slate-400"
        />

        <div className="flex flex-col">
          <button
            type="button"
            disabled={disabled}
            onClick={() => updateValue(numericValue + 1)}
            className="rounded text-slate-500 hover:bg-slate-100 disabled:text-slate-300"
          >
            <ChevronUp className="h-3.5 w-3.5" />
          </button>

          <button
            type="button"
            disabled={disabled}
            onClick={() => updateValue(numericValue - 1)}
            className="rounded text-slate-500 hover:bg-slate-100 disabled:text-slate-300"
          >
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
function Toggle({
  label,
  checked,
  onChange,
  helper,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  helper?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex h-[68px] w-full items-center justify-between rounded-xl border border-slate-200 bg-white px-3 text-left shadow-sm transition hover:border-blue-200"
    >
      <div>
        <div className="text-sm font-semibold text-slate-800">{label}</div>
        {helper ? <div className="mt-0.5 text-xs text-slate-500">{helper}</div> : null}
      </div>
      <span
        className={classNames(
          "relative h-6 w-11 rounded-full transition",
          checked ? "bg-blue-600" : "bg-slate-300",
        )}
      >
        <span
          className={classNames(
            "absolute top-1 h-4 w-4 rounded-full bg-white shadow transition",
            checked ? "left-6" : "left-1",
          )}
        />
      </span>
    </button>
  );
}

function ConfigPanel({
  config,
  setConfig,
  onReset,
}: {
  config: TutorConfig;
  setConfig: React.Dispatch<React.SetStateAction<TutorConfig>>;
  onReset: () => void;
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-3 pb-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-blue-700">
            <Settings2 className="h-4 w-4" /> Runtime configuration
          </div>
          <span className="text-sm font-semibold text-slate-400">Tutor setup</span>
        </div>

        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 transition hover:bg-slate-50"
        >
          <RotateCcw className="h-3.5 w-3.5" /> Reset
        </button>
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}>
        <SelectField
          label="STT backend"
          value={config.sttBackend}
          onChange={(sttBackend) => setConfig((c) => ({ ...c, sttBackend }))}
          options={STT_BACKENDS}
        />

        <SelectField
          label="LLM model"
          value={config.llmModel}
          onChange={(llmModel) => setConfig((c) => ({ ...c, llmModel }))}
          options={LLM_MODELS}
        />

        <SelectField
          label="TTS backend"
          value={config.ttsBackend}
          onChange={(ttsBackend) => setConfig((c) => ({ ...c, ttsBackend }))}
          options={TTS_BACKENDS}
        />

        <SelectField
          label="Teaching style"
          value={config.teachingStyle}
          onChange={(teachingStyle) => setConfig((c) => ({ ...c, teachingStyle }))}
          options={TEACHING_STYLES}
        />

        <Toggle
          label="Use RAG"
          helper="Slide context"
          checked={config.ragEnabled}
          onChange={(ragEnabled) => setConfig((c) => ({ ...c, ragEnabled }))}
        />

        <NumberField
          label="RAG top-k"
          value={config.ragTopK}
          disabled={!config.ragEnabled}
          onChange={(value) =>
            setConfig((c) => ({ ...c, ragTopK: Math.max(1, Number(value || 1)) }))
          }
        />

        <Toggle
          label="Context"
          helper="Previous turns"
          checked={config.conversationContext}
          onChange={(conversationContext) =>
            setConfig((c) => ({ ...c, conversationContext }))
          }
        />

        <NumberField
          label="Max turns"
          value={config.conversationMaxTurns}
          disabled={!config.conversationContext}
          onChange={(conversationMaxTurns) =>
            setConfig((c) => ({ ...c, conversationMaxTurns }))
          }
        />
      </div>

      {/* Qwen3-TTS speaker and language — shown only when qwen3 is selected */}
      {config.ttsBackend === "qwen3" && (
        <div className="mt-3 grid grid-cols-2 gap-3 rounded-2xl border border-violet-100 bg-violet-50/60 p-3">
          <div className="col-span-2 flex items-center gap-2 text-xs font-semibold text-violet-600">
            <Volume2 className="h-3.5 w-3.5" />
            Qwen3-TTS Voice Settings
          </div>
          <SelectField
            label="Speaker"
            value={config.qwen3Speaker}
            onChange={(qwen3Speaker) => setConfig((c) => ({ ...c, qwen3Speaker }))}
            options={QWEN3_SPEAKERS}
          />
          <SelectField
            label="Language"
            value={config.qwen3Language}
            onChange={(qwen3Language) => setConfig((c) => ({ ...c, qwen3Language }))}
            options={QWEN3_LANGUAGES}
          />
        </div>
      )}
    </section>
  );
}


function StagePills({ stage }: { stage: Stage }) {
  return (
    <div className="flex flex-wrap gap-2">
      {stageOrder.map((item) => (
        <div
          key={item}
          className={classNames(
            "rounded-full border px-3 py-1 text-xs font-semibold transition",
            stage === item
              ? "border-blue-200 bg-blue-600 text-white shadow-sm"
              : "border-slate-200 bg-white text-slate-500",
          )}
        >
          {stageLabel[item]}
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={classNames("flex", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={classNames(
          "max-w-[78%] rounded-3xl px-5 py-4 shadow-sm",
          isUser
            ? "rounded-br-md bg-blue-600 text-white"
            : "rounded-bl-md border border-slate-200 bg-white text-slate-900",
        )}
      >
        <div className={classNames("mb-1 text-xs font-semibold", isUser ? "text-blue-100" : "text-slate-500")}>
          {isUser ? "You" : "Tutor"} · {message.timestamp}
        </div>
        <div className="whitespace-pre-wrap text-sm leading-7">{message.text}</div>
        {message.streaming ? (
          <div className="mt-3 flex items-center gap-1 text-blue-500">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current delay-100" />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current delay-200" />
            <span className="ml-2 text-xs text-slate-400">streaming</span>
          </div>
        ) : null}
      </div>
    </motion.div>
  );
}

function LiveUserTranscriptBubble({
  partialText,
  capturedText,
  listening,
}: {
  partialText: string;
  capturedText: string;
  listening: boolean;
}) {
  const visible = Boolean(partialText.trim());

  if (!visible) return null;

  const textToShow = partialText.trim();

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-end"
    >
      <div className="max-w-[78%] rounded-3xl rounded-br-md bg-blue-600 px-5 py-4 text-white shadow-sm">
        <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-blue-100">
          <span>You · live transcription</span>
          {listening ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-white/15 px-2 py-0.5 text-[11px]">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />
              mic on
            </span>
          ) : null}
        </div>

        <div className="whitespace-pre-wrap text-sm leading-7">
          {textToShow}
        </div>
      </div>
    </motion.div>
  );
}

function ConversationPanel({
  messages,
  partialText,
  capturedText,
  listening,
  embedded = false,
}: {
  messages: Message[];
  partialText: string;
  capturedText: string;
  listening: boolean;
  embedded?: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    el.scrollTo({
      top: el.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, partialText, capturedText]);

  return (
    <section
      className={classNames(
        embedded ? "flex min-h-0 flex-1 flex-col overflow-hidden" : "flex h-[82vh] min-h-0 flex-col",
        !embedded && "rounded-3xl border border-slate-200 bg-white shadow-sm"
      )}
    >
      <div className="border-b border-slate-100 px-6 py-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-blue-700">
              <BookOpen className="h-4 w-4" /> Conversation
            </div>
            <h2 className="mt-1 text-xl font-bold text-slate-900">
              Voice-first tutoring session
            </h2>
          </div>
          <StagePills stage={listening ? "listening" : "idle"} />
        </div>
      </div>

      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto bg-slate-50/70 px-6 py-5"
      >
        <div className="space-y-4">
          {messages.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-blue-200 bg-blue-50 p-6 text-blue-950">
              <div className="text-lg font-bold">Start a tutoring conversation</div>
              <p className="mt-2 max-w-2xl text-sm leading-7 text-blue-900">
                Ask by voice or text. Your message appears on the right. The tutor answer streams on the left.
              </p>
            </div>
          ) : null}



          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          <LiveUserTranscriptBubble
            partialText={partialText}
            capturedText={capturedText}
            listening={listening}
          />

          <div />
        </div>
      </div>
    </section>
  );
}

function ControlDock({
  connected,
  listening,
  stage,
  onStartMic,
  onStop,
  onSendText,
  embedded = false,
}: {
  connected: boolean;
  listening: boolean;
  stage: Stage;
  onStartMic: () => void;
  onStop: () => void;
  onSendText: (text: string) => void;
  embedded?: boolean;
}) {
  const [text, setText] = useState("");
  const busy = stage !== "idle" && stage !== "error";

  return (
    <section
      className={classNames(
        "bg-white p-5",
        embedded && "border-t border-slate-100",
        !embedded && "rounded-3xl border border-slate-200 shadow-sm"
      )}
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <button
          type="button"
          disabled={!connected}
          onClick={onStartMic}
          className={classNames(
            "inline-flex h-14 min-w-14 items-center justify-center rounded-2xl font-semibold shadow-sm transition",
            !connected
              ? "cursor-not-allowed bg-slate-200 text-slate-400"
              : listening
                ? "bg-emerald-600 text-white hover:bg-emerald-700"
                : "bg-blue-600 text-white hover:bg-blue-700",
          )}
        >
          <Mic className="h-5 w-5" />
        </button>

        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && text.trim()) {
              onSendText(text.trim());
              setText("");
            }
          }}
          placeholder="Type a question, or use the microphone..."
          className="h-14 flex-1 rounded-2xl border border-slate-200 bg-slate-50 px-4 text-sm text-slate-900 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        />

        <button
          type="button"
          onClick={() => {
            if (!text.trim()) return;
            onSendText(text.trim());
            setText("");
          }}
          className="inline-flex h-14 items-center justify-center gap-2 rounded-2xl bg-slate-900 px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
        >
          <Send className="h-4 w-4" /> Send
        </button>

        <button
          type="button"
          onClick={onStop}
          disabled={!busy && !listening}
          className="inline-flex h-14 items-center justify-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-5 text-sm font-semibold text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
        >
          <Square className="h-4 w-4" /> Stop
        </button>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className="font-semibold text-slate-700">Status:</span>
        <span>{listening ? "Mic on: continuous listening" : stageLabel[stage]}</span>
        <span className="text-slate-300">•</span>
        <span>Stop button turns off the mic and interrupts the current answer/TTS.</span>
      </div>
    </section>
  );
}

function MetricsTable({ history }: { history: TurnMetricRecord[] }) {
  const groupHeaderClass =
    "border-b border-slate-200 px-4 py-3 text-center text-xs font-bold uppercase tracking-wide";
  const columnHeaderClass =
    "px-4 py-3 text-left text-[11px] font-bold uppercase tracking-wide text-slate-500";
  const cellClass =
    "px-4 py-3 text-sm font-medium text-slate-700";
  const numericCellClass =
    "px-4 py-3 text-sm font-semibold tabular-nums text-slate-800";

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-blue-700">
          <Clock3 className="h-4 w-4" /> Latency metrics by turn
        </div>

        <div className="rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
          {history.length} turns recorded
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full min-w-[1180px] border-separate border-spacing-0 text-sm">
            <thead className="sticky top-0 z-20 bg-white">
              <tr>
                <th className="w-[90px] border-b border-slate-200 bg-slate-50 px-4 py-3" />

                <th
                  colSpan={1}
                  className={`${groupHeaderClass} bg-blue-50 text-blue-700`}
                >
                  STT
                </th>

                <th
                  colSpan={3}
                  className={`${groupHeaderClass} bg-sky-50 text-sky-700`}
                >
                  RAG / Context
                </th>

                <th
                  colSpan={2}
                  className={`${groupHeaderClass} bg-indigo-50 text-indigo-700`}
                >
                  LLM
                </th>

                <th
                  colSpan={2}
                  className={`${groupHeaderClass} bg-violet-50 text-violet-700`}
                >
                  TTS
                </th>

                <th
                  colSpan={1}
                  className={`${groupHeaderClass} bg-slate-50 text-slate-700`}
                >
                  Total
                </th>
              </tr>

              <tr className="border-b border-slate-200 bg-slate-50">
                <th className={`${columnHeaderClass} sticky left-0 z-30 bg-slate-50`}>
                  Turn
                </th>
                <th className={columnHeaderClass}>STT latency</th>
                <th className={columnHeaderClass}>RAG latency</th>
                <th className={columnHeaderClass}>Prompt chars</th>
                <th className={columnHeaderClass}>Conversation chars</th>
                <th className={columnHeaderClass}>LLM TTFT</th>
                <th className={columnHeaderClass}>LLM total</th>
                <th className={columnHeaderClass}>TTS first chunk</th>
                <th className={columnHeaderClass}>First audio</th>
                <th className={columnHeaderClass}>Total pipeline</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100">
              {history.length === 0 ? (
                <tr>
                  <td
                    colSpan={10}
                    className="px-4 py-8 text-center text-sm font-medium text-slate-400"
                  >
                    Metrics will appear after the first completed turn.
                  </td>
                </tr>
              ) : (
                history.map((record) => (
                  <tr
                    key={record.turn}
                    className="bg-white transition hover:bg-blue-50/40"
                  >
                    <td className="sticky left-0 z-10 bg-white px-4 py-3 text-sm font-bold text-blue-700">
                      Turn {record.turn}
                    </td>

                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.stt)}
                    </td>

                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.ragLatency)}
                    </td>

                    <td className={cellClass}>
                      {record.metrics.promptChars?.toLocaleString() ?? "—"}
                    </td>

                    <td className={cellClass}>
                      {record.metrics.conversationChars?.toLocaleString() ?? "—"}
                    </td>

                    <td className="px-4 py-3 text-sm font-bold tabular-nums text-blue-700">
                      {formatSeconds(record.metrics.ttft)}
                    </td>

                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.llmTotal)}
                    </td>

                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.ttsFirstChunk)}
                    </td>

                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.firstAudio)}
                    </td>

                    <td className="px-4 py-3 text-sm font-bold tabular-nums text-slate-900">
                      {formatSeconds(record.metrics.total)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function LatencyCharts({ metrics, history }: { metrics: MetricState; history: TurnMetricRecord[] }) {
  const sttValue = metrics.stt ?? 0;
  const llmValue = metrics.ttft ?? 0;
  const ttsValue = metrics.ttsFirstChunk ?? 0;

  const pieData = [
    { name: "STT", value: sttValue },
    { name: "LLM", value: llmValue },
    { name: "TTS", value: ttsValue },
  ].filter((x) => x.value > 0);

  const ttftData = history
    .map((item) => ({
      turn: item.turn,
      ttft: item.metrics.ttft ?? 0,
      contextChars: item.metrics.promptChars ?? item.metrics.conversationChars ?? 0,
    }))
    .filter((x) => x.ttft > 0 && x.contextChars > 0);
  return (
    <section className="flex h-full min-h-0 flex-col rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-blue-700">
        <BarChart3 className="h-4 w-4" /> Latency visualization
      </div>
      <div className="grid min-h-0 flex-1 grid-rows-[0.95fr_1.25fr] gap-4">
        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-100 bg-slate-50 p-3">
          <div className="mb-3 font-semibold text-slate-900">STT / LLM / TTS share</div>
          <div className="mx-auto h-[260px] w-[260px]">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={3}>
                  {pieData.map((_, index) => (
                    <Cell key={index} fill={["#2563eb", "#7c3aed", "#f97316"][index % 3]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${Number(value).toFixed(3)} s`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-100 bg-slate-50 p-3">
          <div className="mb-3 font-semibold text-slate-900">LLM TTFT vs conversation context</div>
          <div className="relative mx-auto aspect-square h-[430px] w-[430px] max-h-full max-w-full">
            {ttftData.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center rounded-2xl border border-dashed border-slate-200 text-center text-sm font-medium text-slate-400">
                One point will appear after each completed turn.
              </div>
            ) : null}

            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={ttftData}
                margin={{ top: 18, right: 22, bottom: 34, left: 22 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />

                <XAxis
                  dataKey="contextChars"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  height={44}
                  axisLine={{ stroke: "#94a3b8", strokeWidth: 1.4 }}
                  tickLine={{ stroke: "#94a3b8" }}
                  label={{
                    value: "context length / prompt chars",
                    position: "insideBottom",
                    offset: -12,
                    fontSize: 11,
                    fill: "#64748b",
                  }}
                />

                <YAxis
                  dataKey="ttft"
                  type="number"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  width={52}
                  axisLine={{ stroke: "#94a3b8", strokeWidth: 1.4 }}
                  tickLine={{ stroke: "#94a3b8" }}
                  label={{
                    value: "LLM TTFT (s)",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    fontSize: 11,
                    fill: "#64748b",
                  }}
                />

                <Tooltip
                  formatter={(value, name) => [
                    name === "ttft" ? `${Number(value).toFixed(3)} s` : value,
                    name === "ttft" ? "LLM TTFT" : name,
                  ]}
                  labelFormatter={(label) => `Context length: ${label} chars`}
                />

                <Line
                  type="monotone"
                  dataKey="ttft"
                  name="LLM TTFT"
                  stroke="#2563eb"
                  strokeWidth={3}
                  dot={{
                    r: 5,
                    stroke: "#2563eb",
                    strokeWidth: 2,
                    fill: "#ffffff",
                  }}
                  activeDot={{
                    r: 7,
                    stroke: "#1d4ed8",
                    strokeWidth: 2,
                    fill: "#2563eb",
                  }}
                  connectNulls
                  isAnimationActive
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}

function SourcesPanel({ sources }: { sources: SourceItem[] }) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-blue-700">
        <Search className="h-4 w-4" /> Retrieved slide context
      </div>
      <div className="space-y-3">
        {sources.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
            Retrieved slides will appear here when RAG is enabled.
          </div>
        ) : (
          sources.map((source) => (
            <div key={source.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="font-semibold text-slate-900">{source.title}</div>
              {source.subtitle ? <div className="mt-1 text-xs font-semibold text-blue-700">{source.subtitle}</div> : null}
              {source.snippet ? <p className="mt-2 text-sm leading-6 text-slate-600">{source.snippet}</p> : null}
              {typeof source.score === "number" ? <div className="mt-2 text-xs text-slate-500">Score: {source.score.toFixed(4)}</div> : null}
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default function Page() {
  const {
    connected,
    stage,
    messages,
    sources,
    metrics,
    metricHistory,
    setMetrics,
    capturedText,
    partialText,
    error,
    send,
    resetTurnMetrics,
    setStage,
    setMessages,
    setCapturedText,
    setPartialText,
    setError,
  } = useTutorSocket("ws://127.0.0.1:8001/ws");

  const [listening, setListening] = useState(false);
  const [config, setConfig] = useState<TutorConfig>(defaultConfig);

  const shouldKeepListeningRef = useRef(false);
  const assistantOutputActiveRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pcmFrameBufferRef = useRef<number[]>([]);
  const stageRef = useRef<Stage>("idle");

  useEffect(() => {
    stageRef.current = stage;

    if (
      stage === "retrieving" ||
      stage === "thinking" ||
      stage === "streaming" ||
      stage === "speaking"
    ) {
      assistantOutputActiveRef.current = true;
    }

    if (
      stage === "idle" ||
      stage === "listening" ||
      stage === "interrupted" ||
      stage === "error"
    ) {
      assistantOutputActiveRef.current = false;
    }
  }, [stage]);

  const backendConfig = useMemo(() => toBackendConfig(config), [config]);
  const isAnswerActive = (currentStage: Stage) =>
    currentStage === "retrieving" ||
    currentStage === "thinking" ||
    currentStage === "streaming" ||
    currentStage === "speaking";

  const downsampleTo16k = (
    input: Float32Array,
    inputSampleRate: number,
    outputSampleRate = 16000,
  ): Float32Array => {
    if (inputSampleRate === outputSampleRate) return input;

    const ratio = inputSampleRate / outputSampleRate;
    const outputLength = Math.floor(input.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.floor((i + 1) * ratio);

      let sum = 0;
      let count = 0;

      for (let j = start; j < end && j < input.length; j++) {
        sum += input[j];
        count++;
      }

      output[i] = count > 0 ? sum / count : 0;
    }

    return output;
  };

  const floatToPcm16Array = (input: Float32Array): number[] => {
    const out: number[] = [];

    for (let i = 0; i < input.length; i++) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      out.push(sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767));
    }

    return out;
  };

  const sendPcm16Frames = (pcm16: number[]) => {
    pcmFrameBufferRef.current.push(...pcm16);

    const frameSize = 320; // 20 ms at 16 kHz

    while (pcmFrameBufferRef.current.length >= frameSize) {
      const frame = pcmFrameBufferRef.current.slice(0, frameSize);
      pcmFrameBufferRef.current = pcmFrameBufferRef.current.slice(frameSize);

      send({
        action: "audio_frame",
        data: frame,
      });
    }
  };

  const sendText = (
    text: string,
    options?: {
      sttLatency?: number;
      addUserMessage?: boolean;
      keepCapturedText?: boolean;
    }
  ) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    resetTurnMetrics();
    const ok = send({ action: "text_query", text: trimmed, config: backendConfig });
    if (!ok) {
      setStage("error");
      setError("Could not send text. Backend websocket is disconnected.");
      return;
    }

    if (typeof options?.sttLatency === "number" && options.sttLatency > 0) {
      setMetrics((prev) => ({
        ...prev,
        stt: options.sttLatency,
      }));
    }

    if (!options?.keepCapturedText) {
      setCapturedText("");
    }

    setPartialText("");

    if (options?.addUserMessage !== false) {
      setMessages((prev) => [
        ...prev,
        {
          id: createId("user_text"),
          role: "user",
          text: trimmed,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    }

    setStage("thinking");
  };


  const stopMic = () => {
    shouldKeepListeningRef.current = false;
    pcmFrameBufferRef.current = [];

    try {
      workletNodeRef.current?.disconnect();
    } catch {
      // ignore
    }

    try {
      sourceNodeRef.current?.disconnect();
    } catch {
      // ignore
    }

    try {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    } catch {
      // ignore
    }

    try {
      void audioContextRef.current?.close();
    } catch {
      // ignore
    }

    workletNodeRef.current = null;
    sourceNodeRef.current = null;
    mediaStreamRef.current = null;
    audioContextRef.current = null;

    send({ action: "stop_audio" });

    setListening(false);
    setCapturedText("");
    setPartialText("");

    if (!isAnswerActive(stageRef.current)) {
      setStage("idle");
    }
  };

  const startMic = async () => {
    if (listening || shouldKeepListeningRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      const audioContext = new AudioContext();
      await audioContext.audioWorklet.addModule("/pcm-worklet.js");

      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "pcm-worklet");

      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;

      source.connect(workletNode);
      workletNode.connect(silentGain);
      silentGain.connect(audioContext.destination);

      workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const floatSamples = event.data;
        const samples16k = downsampleTo16k(floatSamples, audioContext.sampleRate, 16000);
        const pcm16 = floatToPcm16Array(samples16k);
        sendPcm16Frames(pcm16);
      };

      mediaStreamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceNodeRef.current = source;
      workletNodeRef.current = workletNode;
      pcmFrameBufferRef.current = [];

      shouldKeepListeningRef.current = true;


      const ok = send({
        action: "start_stream",
        config: backendConfig,
      });

      if (!ok) {
        stopMic();
        setStage("error");
        setError("Could not start microphone. Backend websocket is disconnected.");
        return;
      }

      setListening(true);
      setCapturedText("");
      setPartialText("");
      setError(null);
      setStage("listening");
    } catch (error) {
      console.error("[MIC] Could not start echo-cancelled microphone stream:", error);
      setListening(false);
      setStage("error");
      setError("Could not start microphone stream.");
    }
  };

  const stopPipeline = () => {
    stopMic();

    assistantOutputActiveRef.current = false;

    send({ action: "interrupt" });

    setCapturedText("");
    setPartialText("");
    setStage("interrupted");
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 via-blue-50 to-white text-slate-900">
      <div className="flex min-h-screen w-full flex-col gap-3 p-3 xl:p-3">
        <header className="rounded-3xl border border-slate-200 bg-white/85 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-bold text-blue-700">
                <Cpu className="h-4 w-4" /> LLM Tutor System
              </div>
              <h1 className="mt-1 text-3xl font-extrabold tracking-tight text-slate-950">
                Real-time voice tutor dashboard
              </h1>

            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={classNames(
                  "rounded-full border px-3 py-1 text-xs font-bold",
                  connected
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                    : "border-red-200 bg-red-50 text-red-700"
                )}
              >
                Backend: {connected ? "Online" : "Offline"}
              </span>

              <span className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700">
                {config.llmModel}
              </span>
              <span className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1 text-xs font-bold text-orange-700">
                TTS: {config.ttsBackend}
              </span>
              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700">
                {config.ragEnabled ? `RAG top-${config.ragTopK}` : "RAG off"}
              </span>
            </div>
          </div>
        </header>

        <div className="flex flex-col gap-5">
          <ConfigPanel
            config={config}
            setConfig={setConfig}
            onReset={() => setConfig(defaultConfig)}
          />

          <div className="grid h-[calc(100vh-1.5rem)] max-h-[calc(100vh-1.5rem)] min-h-0 gap-3 xl:grid-cols-[minmax(0,1.55fr)_minmax(640px,0.82fr)]">
            <div className="flex h-full min-h-0 flex-col">
              <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
                <ConversationPanel
                  messages={messages}
                  partialText={partialText}
                  capturedText={capturedText}
                  listening={listening}
                  embedded
                />

                <ControlDock
                  connected={connected}
                  listening={listening}
                  stage={stage}
                  onStartMic={startMic}
                  onStop={stopPipeline}
                  onSendText={sendText}
                  embedded
                />
              </section>

              <AnimatePresence>
                {error ? (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 8 }}
                    className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700"
                  >
                    {error}
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>

            <LatencyCharts metrics={metrics} history={metricHistory} />
          </div>

          <MetricsTable history={metricHistory} />

          <SourcesPanel sources={sources} />
        </div>
      </div>
    </div>


  );
}
