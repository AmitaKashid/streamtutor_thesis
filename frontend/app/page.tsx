"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Check,
  Lightbulb,
  Clock3,
  Cpu,
  Mic,
  RotateCcw,
  Search,
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
import { idleResponse, responseActivity, responseIsActive } from "./response-activity.mjs";


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
  reasoning?: string;
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
  thinkingTime?: number | null;
  llmAnswerTime?: number | null;
  model?: string | null;
  ttsFirstChunk?: number;
  llmTotal?: number;
  ttsTotal?: number;
  firstAudio?: number;
  total?: number;
  promptChars?: number;
  promptWords?: number;
  conversationTurns?: number;
  conversationChars?: number;
  serverPromptTokens?: number;
  serverCachedTokens?: number;
  serverProcessedPromptTokens?: number;
  serverPrefillSeconds?: number;
  serverOutputTokens?: number;
  serverDecodeSeconds?: number;
  serverDecodeTokensPerSecond?: number;
}

interface TurnMetricRecord {
  turn: number;
  metrics: MetricState;
}



interface TutorConfig {
  sttBackend: string;
  llmModel: string;
  ttsBackend: string;
  ttsEnabled: boolean;
  thinkingEnabled: boolean;
  thinkingBudget: "default" | "256" | "512" | "1024";
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
  | { type: "thinking_chunk"; turn_id: string; text: string }
  | { type: "thinking_preview"; turn_id: string; seconds: number }
  | { type: "llm_answer_progress"; turn_id: string; seconds: number }
  | { type: "assistant_done"; turn_id?: string }
  | { type: "sources"; data: SourceItem[] }
  | { type: "metrics"; turn_id: string; data: MetricState }
  | { type: "captured_text"; text: string }
  | { type: "transcript_partial"; text: string }
  | { type: "transcript_final"; text: string }
  | { type: "user_transcript_final"; text: string }
  | { type: "barge_in_accepted"; text: string }
  | { type: "tutor_turn_started"; turn_id: string; delivery_mode: "audio" | "text"; model?: string | null }
  | { type: "text_interrupted"; turn_id: string }
  | { type: "audio_chunk"; turn_id: string; data: string; sample_rate: number; channels: number }
  | { type: "audio_done"; turn_id: string }
  | { type: "text_complete"; turn_id: string; text: string }
  | { type: "audio_stop"; turn_id: string }
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
  { value: "gemma4_e4b_thinking", label: "Gemma 4 E4B Thinking" },
  { value: "bonsai2_9b_thinking", label: "Bonsai 2 9B Thinking" },
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



const defaultConfig: TutorConfig = {
  sttBackend: "faster-whisper",
  llmModel: "llama3_8b_q4",
  ttsBackend: "kokoro",
  ttsEnabled: true,
  thinkingEnabled: false,
  thinkingBudget: "default",
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

function formatSeconds(value?: number | null) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return `${value.toFixed(3)} s`;
}

function toBackendConfig(config: TutorConfig) {
  return {
    stt_backend: config.sttBackend,
    llm_model: config.llmModel,
    tts_backend: config.ttsBackend,
    tts_enabled: config.ttsEnabled,
    thinking_enabled: config.thinkingEnabled,
    reasoning_budget_tokens: config.thinkingEnabled && config.thinkingBudget !== "default" ? Number(config.thinkingBudget) : null,
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
  const [responseState, setResponseState] = useState(idleResponse);
  const responseStateRef = useRef(responseState);
  const updateResponse = (event: { type: string; audio?: boolean }) => {
    const next = responseActivity(responseStateRef.current, event);
    responseStateRef.current = next;
    setResponseState(next);
  };
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [metrics, setMetrics] = useState<MetricState>({});
  const [metricHistory, setMetricHistory] = useState<TurnMetricRecord[]>([]);
  const [capturedText, setCapturedText] = useState("");
  const textDeliveryNotifiedRef = useRef(new Set<string>());
  const [pendingTextDelivery, setPendingTextDelivery] = useState<{turnId: string; messageId: string; text: string} | null>(null);
  const [partialText, setPartialText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const currentTurnMetricsRef = useRef<MetricState>({});
  const hasRecordedCurrentTurnRef = useRef(false);


  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackQueueRef = useRef<AudioBuffer[]>([]);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const browserAudioPlayingRef = useRef(false);
  const playbackTurnIdRef = useRef<string | null>(null);
  const streamEndedRef = useRef(false);
  const receivedAudioRef = useRef(false);
  const notifiedTurnIdsRef = useRef(new Set<string>());
  const stoppedRef = useRef(false);
  const deliveryModeRef = useRef<"audio" | "text">("audio");

  // Acknowledge only after React has committed the exact completed answer.
  useEffect(() => {
    if (!pendingTextDelivery) return;
    const rendered = messages.some((m) => m.id === pendingTextDelivery.messageId && m.text === pendingTextDelivery.text && !m.streaming);
    const ws = wsRef.current;
    if (!rendered || !ws || ws.readyState !== WebSocket.OPEN || stoppedRef.current || playbackTurnIdRef.current !== pendingTextDelivery.turnId || textDeliveryNotifiedRef.current.has(pendingTextDelivery.turnId)) return;
    textDeliveryNotifiedRef.current.add(pendingTextDelivery.turnId);
    ws.send(JSON.stringify({ action: "browser_text_finished", turn_id: pendingTextDelivery.turnId }));
  }, [messages, pendingTextDelivery]);

  // Audio is complete only after the server ends the stream and browser playback drains.
  const markAudioDrained = () => {
    if (streamEndedRef.current && !currentSourceRef.current && playbackQueueRef.current.length === 0) {
      updateResponse({ type: "audio_done" });
    }
  };

  // Report only when audio_done has arrived AND all browser buffers ended.
  const maybeAcknowledgePlayback = () => {
    const turnId = playbackTurnIdRef.current;
    if (!turnId || stoppedRef.current || !streamEndedRef.current ||
        !receivedAudioRef.current || currentSourceRef.current ||
        playbackQueueRef.current.length > 0 ||
        notifiedTurnIdsRef.current.has(turnId)) return;
    const ws = wsRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return;
    notifiedTurnIdsRef.current.add(turnId);
    ws.send(JSON.stringify({ action: "browser_audio_finished", turn_id: turnId }));
  };

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

        markAudioDrained();
        maybeAcknowledgePlayback();
      }
    };

    source.start();
  };

  const stopBrowserAudio = (turnId?: string) => {
    const id = turnId || playbackTurnIdRef.current;
    stoppedRef.current = true;
    streamEndedRef.current = false;
    playbackQueueRef.current = [];
    const source = currentSourceRef.current;
    // Remove before calling stop; stop() triggers onended asynchronously.
    currentSourceRef.current = null;
    if (source) source.onended = null;
    try { source?.stop(); } catch { /* already stopped */ }
    browserAudioPlayingRef.current = false;
    const ws = wsRef.current;
    if (deliveryModeRef.current === "audio" && id && !notifiedTurnIdsRef.current.has(id) &&
        ws?.readyState === WebSocket.OPEN) {
      notifiedTurnIdsRef.current.add(id);
      ws.send(JSON.stringify({ action: "browser_audio_interrupted", turn_id: id }));
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
        stopBrowserAudio();
        updateResponse({ type: "reset" });
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
          if (data.type === "tutor_turn_started") {
            // Never mark the prior turn delivered when a new turn starts.
            if (playbackTurnIdRef.current &&
                playbackTurnIdRef.current !== data.turn_id) {
              stopBrowserAudio(playbackTurnIdRef.current);
            }
            playbackTurnIdRef.current = data.turn_id;
            updateResponse({ type: "start", audio: data.delivery_mode === "audio" });
            const turnModel = data.model ?? null;
            // Each turn starts with a clean record, including voice and barge-in turns.
            currentTurnMetricsRef.current = { model: turnModel, thinkingTime: null, llmAnswerTime: null };
            hasRecordedCurrentTurnRef.current = false;
            setMetrics({ model: turnModel, thinkingTime: null, llmAnswerTime: null });
            deliveryModeRef.current = data.delivery_mode;
            streamEndedRef.current = false;
            receivedAudioRef.current = false;
            stoppedRef.current = responseStateRef.current.cancelling;
            return;
          }

          if (data.type === "text_interrupted") {
            if (data.turn_id === playbackTurnIdRef.current) {
              stoppedRef.current = true;
              currentTurnMetricsRef.current = { ...currentTurnMetricsRef.current, thinkingTime: null, llmAnswerTime: null };
              setMetrics((prev) => ({ ...prev, thinkingTime: null, llmAnswerTime: null }));
            }
            return;
          }

          if (data.type === "text_complete") {
            if (data.turn_id !== playbackTurnIdRef.current || stoppedRef.current) return;
            const messageId = `assistant_${data.turn_id}`;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              const message: Message = { id: messageId, role: "assistant", text: data.text, reasoning: last?.role === "assistant" && last.streaming ? last.reasoning : undefined, streaming: false, timestamp: last?.timestamp || new Date().toLocaleTimeString() };
              if (last?.role === "assistant" && last.streaming) next[next.length - 1] = message;
              else next.push(message);
              return next;
            });
            setPendingTextDelivery({ turnId: data.turn_id, messageId, text: data.text });
            return;
          }

          if (data.type === "audio_chunk") {
            if (data.turn_id !== playbackTurnIdRef.current || stoppedRef.current) return;
            receivedAudioRef.current = true;
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
            if (data.turn_id === playbackTurnIdRef.current) {
              stopBrowserAudio(data.turn_id);
              if (responseStateRef.current.generating) {
                currentTurnMetricsRef.current = { ...currentTurnMetricsRef.current, thinkingTime: null, llmAnswerTime: null };
                setMetrics((prev) => ({ ...prev, thinkingTime: null, llmAnswerTime: null }));
              }
              updateResponse({ type: "audio_done" });
            }
            return;
          }

          if (data.type === "audio_done") {
            if (data.turn_id !== playbackTurnIdRef.current || stoppedRef.current) return;
            streamEndedRef.current = true;
            markAudioDrained();
            maybeAcknowledgePlayback();
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

          if (data.type === "thinking_preview" || data.type === "llm_answer_progress") {
            if (data.turn_id !== playbackTurnIdRef.current || stoppedRef.current ||
                !Number.isFinite(data.seconds) || data.seconds < 0) return;
            const update: MetricState = data.type === "thinking_preview"
              ? { thinkingTime: data.seconds } : { llmAnswerTime: data.seconds };
            currentTurnMetricsRef.current = { ...currentTurnMetricsRef.current, ...update };
            setMetrics((prev) => ({ ...prev, ...update }));
            return;
          }

          if (data.type === "thinking_chunk") {
            if (data.turn_id !== playbackTurnIdRef.current || stoppedRef.current) return;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant" && last.streaming) {
                next[next.length - 1] = { ...last, reasoning: (last.reasoning || "") + data.text };
              } else {
                next.push({ id: `assistant_${data.turn_id}`, role: "assistant", text: "", reasoning: data.text, streaming: true, timestamp: new Date().toLocaleTimeString() });
              }
              return next;
            });
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
            // Completion from an interrupted older turn must not end a newer turn.
            if (data.turn_id && playbackTurnIdRef.current && data.turn_id !== playbackTurnIdRef.current) return;
            if (data.turn_id && !playbackTurnIdRef.current && !responseStateRef.current.cancelling) return;
            updateResponse({ type: "generation_done" });
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
            // Late metrics from a stopped or superseded turn must not affect this row.
            if (!data.turn_id || data.turn_id !== playbackTurnIdRef.current) return;
            const incoming = data.data;

            // Record immediately: assistant_done can arrive before React runs
            // a queued state updater, especially for short text-only turns.
            const previousTurn = currentTurnMetricsRef.current;
            currentTurnMetricsRef.current = {
              ...previousTurn,
              ...incoming,
              stt: incoming.stt !== undefined && incoming.stt > 0
                ? incoming.stt
                : previousTurn.stt,
            };
            setMetrics((prev) => ({
              ...prev,
              ...incoming,
              stt: incoming.stt !== undefined && incoming.stt > 0
                ? incoming.stt
                : prev.stt,
            }));

            return;
          }

          if (data.type === "error") {
            currentTurnMetricsRef.current = { ...currentTurnMetricsRef.current, thinkingTime: null, llmAnswerTime: null };
            setMetrics((prev) => ({ ...prev, thinkingTime: null, llmAnswerTime: null }));
            stopBrowserAudio();
            updateResponse({ type: "reset" });
            setError(data.message);
            setStage("error");
          }
        } catch {
          stopBrowserAudio();
          updateResponse({ type: "reset" });
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

  const beginResponse = () => {
    if (responseIsActive(responseStateRef.current)) return false;
    playbackTurnIdRef.current = null;
    updateResponse({ type: "submit" });
    return true;
  };

  const stopResponse = () => {
    if (!responseIsActive(responseStateRef.current) || responseStateRef.current.cancelling) return;
    updateResponse({ type: "stop" });
    stopBrowserAudio();
    if (!send({ action: "interrupt" })) updateResponse({ type: "reset" });
  };

  const resetTurnMetrics = () => {
    currentTurnMetricsRef.current = {};
    hasRecordedCurrentTurnRef.current = false;
    setMetrics({});
  };

  return {
    connected,
    stage,
    responseActive: responseIsActive(responseState),
    responseStopping: responseState.cancelling,
    beginResponse,
    stopResponse,
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
          onChange={(llmModel) => setConfig((c) => ({ ...c, llmModel, thinkingEnabled: llmModel === c.llmModel ? c.thinkingEnabled : false }))}
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
      {config.ttsEnabled && config.ttsBackend === "qwen3" && (
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
        {message.reasoning ? (
          <details open={message.streaming && !message.text} className="mb-3 rounded-xl bg-slate-50 p-3 text-slate-600">
            <summary className="cursor-pointer text-xs font-semibold">Thinking</summary>
            <div className="mt-2 whitespace-pre-wrap text-sm leading-6">{message.reasoning}</div>
          </details>
        ) : null}
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
  speechEnabled,
  onSpeechChange,
  embedded = false,
}: {
  messages: Message[];
  partialText: string;
  capturedText: string;
  listening: boolean;
  speechEnabled: boolean;
  onSpeechChange: (enabled: boolean) => void;
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
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-blue-700">
              <BookOpen className="h-4 w-4" /> Conversation
            </div>
            <h2 className="mt-1 text-xl font-bold text-slate-900">
              Voice-first tutoring session
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-2" aria-label="Response options">
            <button
              type="button" role="switch" aria-label="Read aloud" aria-checked={speechEnabled}
              onClick={() => onSpeechChange(!speechEnabled)} title="Read the tutor’s answer aloud; applies to the next turn"
              className="inline-flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-blue-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
            >
              <Volume2 className="h-4 w-4 text-blue-600" aria-hidden="true" />
              <span>Read aloud</span>
              <span aria-hidden="true" className={classNames("relative h-5 w-9 rounded-full transition", speechEnabled ? "bg-blue-600" : "bg-slate-300")}>
                <span className={classNames("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition", speechEnabled ? "left-[18px]" : "left-0.5")} />
              </span>
            </button>
          </div>
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

const THINKING_BUDGETS = [
  { value: "default", label: "Default", detail: "Configured default" },
  { value: "256", label: "Short", detail: "Max 256 tokens" },
  { value: "512", label: "Medium", detail: "Max 512 tokens" },
  { value: "1024", label: "Long", detail: "Max 1,024 tokens" },
] as const;

function ThinkingBudgetControl({ enabled, available, value, onChange }: {
  enabled: boolean;
  available: boolean;
  value: TutorConfig["thinkingBudget"];
  onChange: (enabled: boolean, value: TutorConfig["thinkingBudget"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const selected = THINKING_BUDGETS.find((option) => option.value === value);

  useEffect(() => {
    if (!available) setOpen(false);
  }, [available]);

  useEffect(() => {
    if (!open) return;
    const closeOnOutside = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        buttonRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOnOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const choose = (nextEnabled: boolean, nextValue: TutorConfig["thinkingBudget"]) => {
    onChange(nextEnabled, nextValue);
    setOpen(false);
    buttonRef.current?.focus();
  };

  return (
    <div ref={containerRef} className="relative shrink-0">
      <button ref={buttonRef} type="button" disabled={!available}
        aria-label={available ? `Thinking: ${enabled ? selected?.label ?? "Default" : "Off"}` : "Thinking unavailable for this model"}
        aria-expanded={open} aria-controls={open ? "thinking-budget-popover" : undefined}
        title={available ? "Thinking and reasoning token limit" : "Thinking is unavailable for this model"}
        onClick={() => { if (!enabled) onChange(true, value); setOpen((current) => !current); }}
        className={classNames("inline-flex h-10 items-center gap-1.5 rounded-full px-3 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:opacity-50", enabled && available ? "bg-blue-100 text-blue-700 hover:bg-blue-200" : "text-slate-600 hover:bg-slate-100")}>
        <Lightbulb className="h-4 w-4" aria-hidden="true" />
        <span>{enabled && available ? selected?.label ?? "Default" : "Think"}</span>
        <ChevronDown className="h-4 w-4" aria-hidden="true" />
      </button>
      {open && available && (
        <div id="thinking-budget-popover" role="group" aria-label="Thinking settings"
          className="absolute bottom-full right-0 z-50 mb-3 w-72 max-w-[calc(100vw-3rem)] rounded-2xl border border-slate-200 bg-white p-2 shadow-xl shadow-slate-900/10">
          <div className="px-3 pb-2 pt-2 text-sm font-bold text-slate-900">Thinking</div>
          <button type="button" aria-pressed={!enabled} onClick={() => choose(false, value)}
            className={classNames("flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm text-slate-700 focus-visible:outline-2 focus-visible:outline-blue-600", !enabled ? "bg-blue-50 font-semibold" : "hover:bg-slate-50")}>
            <span className="w-4 shrink-0">{!enabled && <Check className="h-4 w-4 text-blue-600" aria-hidden="true" />}</span><span>Off</span>
          </button>
          {THINKING_BUDGETS.map((option) => {
            const active = enabled && option.value === value;
            return (
              <button key={option.value} type="button" aria-pressed={active} onClick={() => choose(true, option.value)}
                className={classNames("flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm text-slate-700 focus-visible:outline-2 focus-visible:outline-blue-600", active ? "bg-blue-50 font-semibold" : "hover:bg-slate-50")}>
                <span className="w-4 shrink-0">{active && <Check className="h-4 w-4 text-blue-600" aria-hidden="true" />}</span>
                <span>{option.label}</span><span className="ml-auto text-xs font-normal text-slate-500">{option.detail}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ControlDock({
  connected,
  listening,
  stage,
  onStartMic,
  onStop,
  responseActive,
  responseStopping,
  onSendText,
  thinkingEnabled,
  thinkingAvailable,
  thinkingBudget,
  onThinkingSettingsChange,
  embedded = false,
}: {
  connected: boolean;
  listening: boolean;
  stage: Stage;
  onStartMic: () => void;
  onStop: () => void;
  responseActive: boolean;
  responseStopping: boolean;
  onSendText: (text: string) => boolean;
  thinkingEnabled: boolean;
  thinkingAvailable: boolean;
  thinkingBudget: TutorConfig["thinkingBudget"];
  onThinkingSettingsChange: (enabled: boolean, value: TutorConfig["thinkingBudget"]) => void;
  embedded?: boolean;
}) {
  const [text, setText] = useState("");

  return (
    <section
      className={classNames(
        "bg-white p-5",
        embedded && "border-t border-slate-100",
        !embedded && "rounded-3xl border border-slate-200 shadow-sm"
      )}
    >
      <div className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-2 focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-100 md:flex-row md:items-center">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing && !responseActive && connected && text.trim()) {
              e.preventDefault();
              if (onSendText(text.trim())) setText("");
            }
          }}
          placeholder="Type a question and press Enter..."
          className="h-12 min-w-0 flex-1 bg-transparent px-3 text-sm text-slate-900 outline-none"
        />

        <div className="flex items-center justify-end gap-2">
          <ThinkingBudgetControl enabled={thinkingEnabled} available={thinkingAvailable} value={thinkingBudget} onChange={onThinkingSettingsChange} />
          <button
            type="button"
            disabled={!connected}
            onClick={onStartMic}
            aria-label={listening ? "Stop microphone" : "Start microphone"}
            aria-pressed={listening}
            className={classNames(
              "inline-flex h-12 w-12 shrink-0 items-center justify-center self-end rounded-full font-semibold shadow-sm transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 md:self-center",
              !connected
                ? "cursor-not-allowed bg-slate-200 text-slate-400"
                : listening
                  ? "bg-emerald-600 text-white hover:bg-emerald-700"
                  : "bg-blue-600 text-white hover:bg-blue-700",
            )}
          >
            <Mic className="h-5 w-5" />
          </button>

          {responseActive && (
            <button
              type="button"
              aria-label="Stop response"
              title="Stop response"
              disabled={!connected || responseStopping}
              onClick={onStop}
              className="inline-flex h-12 w-12 shrink-0 items-center justify-center self-end rounded-full bg-blue-600 text-white shadow-sm transition hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 md:self-center"
            >
              <Square className="h-4 w-4 fill-current" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className="font-semibold text-slate-700">Status:</span>
        <span>{listening ? "Mic on: continuous listening" : stageLabel[stage]}</span>

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
                  colSpan={10}
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
                <th className={columnHeaderClass} title="Previous conversation included in this request, before the current answer is stored.">History sent</th>
                <th className={columnHeaderClass}>Model</th>
                <th className={columnHeaderClass} title="Full tokenized prompt, including system and chat template; measured by the model server.">Prompt tokens</th>
                <th className={columnHeaderClass} title="Prompt tokens reused from the server's prefix cache.">Cached tokens</th>
                <th className={columnHeaderClass} title="Server prompt-processing time; excludes queue and network. Cached tokens are not reprocessed.">Prefill</th>
                <th className={columnHeaderClass} title="All generated tokens, including thinking when enabled.">Output tokens</th>
                <th className={columnHeaderClass} title="Server-reported generation time, including reasoning and answer tokens.">Decode</th>
                <th className={columnHeaderClass} title="Server-reported generated token rate; includes reasoning when enabled.">Decode tok/s</th>
                <th className={columnHeaderClass} title="First reasoning chunk to first answer chunk; client-observed.">Thinking</th>
                <th className={columnHeaderClass} title="Time until the first answer text is available, including any thinking.">LLM TTFT (answer)</th>
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
                    colSpan={18}
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
                      {record.metrics.conversationChars == null ? "—" : `${record.metrics.conversationChars.toLocaleString()} chars · ${record.metrics.conversationTurns ?? "—"} turns`}
                    </td>

                    <td className={cellClass}>
                      {LLM_MODELS.find((item) => item.value === record.metrics.model)?.label ?? record.metrics.model ?? "—"}
                    </td>
                    <td className={numericCellClass}>{record.metrics.serverPromptTokens?.toLocaleString() ?? "—"}</td>
                    <td className={numericCellClass}>{record.metrics.serverCachedTokens?.toLocaleString() ?? "—"}</td>
                    <td className={numericCellClass}>{formatSeconds(record.metrics.serverPrefillSeconds)}</td>
                    <td className={numericCellClass}>{record.metrics.serverOutputTokens?.toLocaleString() ?? "—"}</td>
                    <td className={numericCellClass}>{formatSeconds(record.metrics.serverDecodeSeconds)}</td>
                    <td className={numericCellClass}>{record.metrics.serverDecodeTokensPerSecond?.toFixed(1) ?? "—"}</td>
                    <td className={numericCellClass}>
                      {formatSeconds(record.metrics.thinkingTime)}
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
  const measuredThinking = metrics.thinkingTime;
  // Thinking is included in first-answer TTFT. Do not double-count it.
  const thinkingValid = measuredThinking != null && Number.isFinite(measuredThinking)
    && measuredThinking >= 0 && (metrics.ttft == null || measuredThinking <= llmValue);
  const thinkingValue = thinkingValid ? measuredThinking : 0;
  const ttsValue = metrics.ttsFirstChunk ?? 0;
  const answerValue = metrics.llmAnswerTime;
  const answerAvailable = answerValue != null && Number.isFinite(answerValue) && answerValue >= 0;

  const latencyShareData = [
    {
      name: "STT",
      description: "Speech-to-text latency",
      available: metrics.stt != null && metrics.stt > 0,
      value: sttValue,
      color: "#2563eb",
    },
    {
      name: "Thinking",
      description: "First reasoning chunk to first answer chunk",
      available: thinkingValid,
      value: thinkingValue,
      color: "#a855f7",
    },
    {
      name: "Other LLM time",
      description: "Time to first answer excluding the thinking interval; includes prefill, network and buffering",
      available: metrics.ttft != null,
      value: Math.max(0, llmValue - thinkingValue),
      color: "#7c3aed",
    },
    {
      name: "LLM answer",
      description: "Elapsed time streaming the answer, including pauses",
      available: answerAvailable,
      value: answerAvailable ? answerValue : 0,
      color: "#0d9488",
    },
    {
      name: "TTS",
      description: "Time from first text chunk to first audio",
      available: metrics.ttsFirstChunk != null,
      value: ttsValue,
      color: "#f97316",
    },
  ];

  const pieData = latencyShareData.filter((item) => item.value > 0);

  const ttftData = history
    .map((item) => ({
      turn: item.turn,
      ttft: item.metrics.ttft ?? 0,
      contextChars: item.metrics.promptChars ?? item.metrics.conversationChars ?? 0,
      model: LLM_MODELS.find((model) => model.value === item.metrics.model)?.label ?? item.metrics.model ?? "—",
    }))
    .filter((x) => x.ttft > 0 && x.contextChars > 0);
  return (
    <section className="flex h-full min-h-0 flex-col rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-blue-700">
        <BarChart3 className="h-4 w-4" /> Latency visualization
      </div>
      <div className="grid min-h-0 flex-1 grid-rows-[0.95fr_1.25fr] gap-4">
        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-blue-50/60 p-4">
          <div className="mb-3 flex items-center gap-2" title="Relative shares of available timings. Components can overlap; thinking is included in time to first answer.">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-100 text-blue-700" aria-hidden="true">
              <BarChart3 className="h-4 w-4" />
            </span>
            <div className="text-sm font-semibold text-slate-900">Latency breakdown</div>
          </div>
          <div className="flex min-h-[260px] flex-1 flex-col items-center justify-center gap-6 rounded-xl border border-slate-100 bg-white/90 px-2 py-3 sm:flex-row">
            <div className="h-[260px] w-[260px] shrink-0">
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={55}
                    outerRadius={90}
                    paddingAngle={3}
                  >
                    {pieData.map((item) => (
                      <Cell key={item.name} fill={item.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${Number(value).toFixed(3)} s`} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div
              className="flex w-full max-w-[270px] flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
              aria-label="Latency chart legend"
            >
              {latencyShareData.map((item) => (
                <div key={item.name} className="flex items-center gap-2" title={item.description}>
                  <span
                    className="h-3 w-3 shrink-0 rounded-full"
                    style={{ backgroundColor: item.color }}
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 text-sm font-semibold text-slate-800">{item.name}</span>
                  <span className="shrink-0 text-sm tabular-nums text-slate-600">{item.available ? formatSeconds(item.value) : "—"}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-2xl border border-slate-200/80 bg-gradient-to-br from-white to-blue-50/60 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-100 text-blue-700" aria-hidden="true">
                <BarChart3 className="h-4 w-4" />
              </span>
              <div className="text-sm font-semibold text-slate-900">LLM TTFT vs conversation context</div>
            </div>
            <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-500 ring-1 ring-slate-200">seconds</span>
          </div>
          <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-100 bg-white/90 px-2 pb-2 pt-3">
            <div className="min-h-0 w-full flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ttftData} margin={{ top: 12, right: 22, bottom: 2, left: 6 }}>
                  <CartesianGrid vertical={false} strokeDasharray="4 5" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="contextChars"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tick={{ fontSize: 11, fill: "#64748b" }}
                    tickLine={false}
                    axisLine={{ stroke: "#cbd5e1" }}
                    tickMargin={9}
                    height={30}
                  />
                  <YAxis
                    dataKey="ttft"
                    type="number"
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 11, fill: "#64748b" }}
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    width={70}
                    label={{
                      value: "LLM TTFT (seconds)",
                      angle: -90,
                      position: "insideLeft",
                      offset: 10,
                      fontSize: 11,
                      fill: "#64748b",
                    }}
                  />
                  <Tooltip
                    cursor={{ stroke: "#93c5fd", strokeDasharray: "4 4", strokeWidth: 1 }}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const point = payload[0]?.payload as { turn: number; ttft: number; contextChars: number; model: string } | undefined;
                      if (!point) return null;
                      return (
                        <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-lg">
                          <div className="font-semibold text-slate-900">Turn {point.turn} · {point.model}</div>
                          <div className="mt-1">TTFT: <span className="font-semibold tabular-nums text-blue-700">{point.ttft.toFixed(3)} s</span></div>
                          <div>Context: <span className="font-medium tabular-nums">{point.contextChars.toLocaleString()} chars</span></div>
                        </div>
                      );
                    }}
                  />
                  <Line
                    type="linear"
                    dataKey="ttft"
                    name="LLM TTFT"
                    stroke="#2563eb"
                    strokeWidth={3}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    dot={{ r: 4.5, stroke: "#2563eb", strokeWidth: 2, fill: "#ffffff" }}
                    activeDot={{ r: 6.5, stroke: "#ffffff", strokeWidth: 2, fill: "#2563eb" }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="pt-1 text-center text-[11px] font-medium text-slate-500">Prompt / context length (characters)</div>
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
    responseActive,
    responseStopping,
    beginResponse,
    stopResponse,
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

  useEffect(() => {
    if (connected) send({ action: "set_speech_output", tts_enabled: config.ttsEnabled });
  // send reads the current socket; changing the switch affects only the next turn.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, config.ttsEnabled]);

  useEffect(() => {
    if (connected) send({ action: "set_thinking", thinking_enabled: config.thinkingEnabled });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, config.thinkingEnabled]);

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
    if (!trimmed || !connected || !beginResponse()) return false;
    resetTurnMetrics();
    const ok = send({ action: "text_query", text: trimmed, config: backendConfig });
    if (!ok) {
      setStage("error");
      setError("Could not send text. Backend websocket is disconnected.");
      stopResponse();
      return false;
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
    return true;
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
    assistantOutputActiveRef.current = false;
    stopResponse();
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
                  speechEnabled={config.ttsEnabled}
                  onSpeechChange={(ttsEnabled) => setConfig((c) => ({ ...c, ttsEnabled }))}
                  embedded
                />

                <ControlDock
                  connected={connected}
                  listening={listening}
                  stage={stage}
                  onStartMic={listening ? stopMic : startMic}
                  onStop={stopPipeline}
                  responseActive={responseActive}
                  responseStopping={responseStopping}
                  onSendText={sendText}
                  thinkingEnabled={config.thinkingEnabled}
                  thinkingAvailable={["gemma4_e4b_thinking", "bonsai2_9b_thinking", "gemma4_26b_q3", "qwen3_5_9b_q6"].includes(config.llmModel)}
                  thinkingBudget={config.thinkingBudget}
                  onThinkingSettingsChange={(thinkingEnabled, thinkingBudget) => setConfig((c) => ({ ...c, thinkingEnabled, thinkingBudget }))}
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
