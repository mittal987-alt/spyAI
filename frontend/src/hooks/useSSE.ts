import { useEffect, useRef, useState, useCallback } from "react";

export type SSEStatus = "connecting" | "connected" | "disconnected" | "error";

export interface SSEEvent {
  type: string;
  [key: string]: any;
}

interface UseSSEOptions {
  onEvent?: (event: SSEEvent) => void;
  enabled?: boolean;
}

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export function useSSE(options: UseSSEOptions = {}) {
  const { onEvent, enabled = true } = options;
  const [status, setStatus] = useState<SSEStatus>("disconnected");
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(1000);

  const connect = useCallback(() => {
    if (!enabled) return;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setStatus("connecting");
    const es = new EventSource(`${API_BASE}/api/v1/events/stream`);
    eventSourceRef.current = es;

    es.onopen = () => {
      setStatus("connected");
      backoffRef.current = 1000;
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastEvent(data);
        if (onEventRef.current) {
          onEventRef.current(data);
        }
      } catch (err) {
        console.warn("Failed to parse SSE event data:", event.data, err);
      }
    };

    es.onerror = () => {
      setStatus("error");
      es.close();
      eventSourceRef.current = null;

      // Exponential backoff reconnect up to 15s
      const delay = Math.min(backoffRef.current, 15000);
      backoffRef.current = Math.min(backoffRef.current * 1.5, 15000);

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(() => {
        if (enabled) {
          connect();
        }
      }, delay);
    };
  }, [enabled]);

  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setStatus("disconnected");
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [connect, enabled]);

  return {
    status,
    isConnected: status === "connected",
    lastEvent,
    reconnect: connect,
  };
}
