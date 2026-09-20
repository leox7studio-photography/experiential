//! Azure annotation-only frames share the compatible content and finish lifecycle,
//! and a finish reason without the `[DONE]` sentinel is still a complete ending.

use crate::dialects::{drain_stream_fixture, Dialect, Normalizer};
use crate::errors::FailureClass;
use crate::events::Event;
use crate::sse::SseEvent;
use serde_json::{json, Value};

fn frame(choice: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [choice]}).to_string(),
    }
}

fn annotation(finish: Option<&str>) -> Value {
    let filtered = finish == Some("content_filter");
    json!({
        "index": 0, "finish_reason": finish,
        "content_filter_results": {
            "hate": {"filtered": filtered, "severity": if filtered { "high" } else { "safe" }}
        },
        "content_filter_offsets": {"check_offset": 49, "start_offset": 47, "end_offset": 49}
    })
}

#[test]
fn azure_annotations_preserve_text_finish_and_trailing_usage() {
    for after_stop in [false, true] {
        let mut frames = vec![
            json!({"choices": [], "prompt_filter_results": []}),
            json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            json!({"choices": [annotation(None)]}),
            json!({"choices": [{"index": 0, "delta": {"content": "OK"}}]}),
        ];
        let stop = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        let annotation = json!({"choices": [annotation(None)]});
        frames.extend(if after_stop {
            vec![stop, annotation]
        } else {
            vec![annotation, stop]
        });
        frames.push(json!({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14}}));
        let wire = frames
            .iter()
            .map(|value| format!("data: {value}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";
        // Decode across arbitrary transport boundaries, including inside JSON.
        let chunks = wire
            .as_bytes()
            .chunks(7)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
        assert!(failure.is_none(), "{failure:?}");
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "OK"}),
                json!({"kind": "usage", "input_tokens": 13, "output_tokens": 1, "cached_input_tokens": null, "reasoning_tokens": null}),
                json!({"kind": "completed"}),
            ]
        );
    }
}

#[test]
fn azure_filter_annotation_preserves_refusal_instead_of_completion() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let events = normalizer
        .feed(&frame(annotation(Some("content_filter"))))
        .unwrap();
    assert!(matches!(events.as_slice(), [Event::RefusalDelta(_)]));
    normalizer.feed(&frame(annotation(None))).unwrap();
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".into(),
        })
        .unwrap();
    assert!(
        matches!(events.as_slice(), [Event::Failed(failure)] if failure.failure_class == FailureClass::Refusal)
    );
}

#[test]
fn compatible_missing_or_invalid_deltas_still_fail_without_valid_annotations() {
    let mut cases = vec![
        json!({"index": 0}),
        json!({"index": 0, "finish_reason": "stop"}),
    ];
    for delta in [Value::Null, json!("bad"), json!([]), json!(0)] {
        let mut choice = annotation(None);
        choice["delta"] = delta;
        cases.push(choice);
    }
    for field in ["content_filter_results", "content_filter_offsets"] {
        for value in [Value::Null, json!("bad"), json!([])] {
            let mut choice = annotation(None);
            choice[field] = value;
            cases.push(choice);
        }
        let mut choice = annotation(None);
        choice.as_object_mut().unwrap().remove(field);
        cases.push(choice);
    }
    for choice in cases {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        assert!(
            normalizer.feed(&frame(choice.clone())).is_err(),
            "accepted {choice}"
        );
    }
}

/// An OpenAI-compatible lane that answers HTTP 200 with an error-shaped body
/// (no `choices`) declares its failure in whichever envelope spelling it
/// uses; each reaches the ledger with its sentence instead of dying as a
/// malformed "choices must be an array" frame.
#[test]
fn error_shaped_success_frames_declare_the_provider_failure_with_detail() {
    for (payload, expected_class, expected_detail) in [
        (
            json!({"object": "error", "message": "Tool 'g' not found in tools list.",
                   "type": "BadRequestError", "param": null, "code": 400}),
            FailureClass::InvalidRequest,
            "400: Tool 'g' not found in tools list.",
        ),
        (
            json!({"code": "invalid-argument", "error": "Argument not supported on this model: presencePenalty"}),
            FailureClass::InvalidRequest,
            "invalid-argument: Argument not supported on this model: presencePenalty",
        ),
        (
            json!({"code": 400, "reason": "INVALID_PARAMETER",
                   "message": "tools is not supported by this model", "metadata": {}}),
            FailureClass::InvalidRequest,
            "INVALID_PARAMETER: tools is not supported by this model",
        ),
        (
            // A FastAPI validation list is the origin refusing the request's
            // shape: its pydantic token classifies it as the caller's error.
            json!({"detail": [{"loc": ["body", "messages", 0, "content"],
                   "msg": "field required", "type": "value_error.missing"}]}),
            FailureClass::InvalidRequest,
            "value_error.missing: field required",
        ),
        (
            // A plain-string detail names no class; it stays the provider's
            // failure with its sentence kept for the ledger.
            json!({"detail": "The origin could not load the model."}),
            FailureClass::ProviderInternal,
            "The origin could not load the model.",
        ),
        (
            // API Management's envelope carries only a status and a sentence.
            json!({"statusCode": 400, "message": "Invalid request body."}),
            FailureClass::InvalidRequest,
            "400: Invalid request body.",
        ),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let events = normalizer
            .feed(&SseEvent {
                event: None,
                data: payload.to_string(),
            })
            .expect("an error-shaped frame is a declared failure, never malformed");
        let failure = match events.as_slice() {
            [Event::Failed(failure)] => failure,
            other => panic!("expected one failed terminal for {payload}, got {other:?}"),
        };
        assert_eq!(failure.failure_class, expected_class, "{payload}");
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some(expected_detail),
            "{payload}"
        );
    }
    // A frame with no `choices` and no error marker is not an envelope: it
    // keeps the strict malformed contract instead of settling as a failure.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let malformed = normalizer
        .feed(&SseEvent {
            event: None,
            data: json!({"message": "warming up"}).to_string(),
        })
        .expect_err("a bare message is not a declared failure");
    assert_eq!(malformed.failure_class, FailureClass::MalformedResponse);
}

fn wire(frames: &[Value], done: bool) -> Vec<Vec<u8>> {
    let mut text = frames
        .iter()
        .map(|value| format!("data: {value}\n\n"))
        .collect::<String>();
    if done {
        text += "data: [DONE]\n\n";
    }
    // Decode across arbitrary transport boundaries, including inside JSON.
    text.as_bytes().chunks(11).map(<[u8]>::to_vec).collect()
}

fn text_frame(content: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]})
}

fn finish_frame(finish: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
}

/// Azure AI Foundry's DeepSeek-V4-Flash deployment (live 2026-09-15) ends a
/// content-filtered stream with the `finish_reason: "content_filter"` chunk
/// and closes the connection without `data: [DONE]`. The finish reason is the
/// provider's declared ending, so the stream settles as the refusal it named
/// instead of "provider stream ended without a terminal event".
#[test]
fn content_filter_finish_without_done_sentinel_is_the_declared_refusal() {
    let frames = [
        json!({"choices": [], "prompt_filter_results": []}),
        text_frame("The court"),
        text_frame(" remanded"),
        finish_frame("content_filter"),
    ];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, false));
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "The court"}),
            json!({"kind": "text_delta", "text": " remanded"}),
            json!({"kind": "refusal_delta", "text": ""}),
            json!({
                "kind": "failed",
                "failure_class": "refusal",
                "safe_message": "provider refused the request: content policy",
                "refusal_reason": "content_policy",
            }),
        ]
    );
}

/// A `stop` finish followed by EOF (no sentinel) completes and folds the
/// trailing usage exactly as the `[DONE]` path does.
#[test]
fn stop_finish_without_done_sentinel_completes_with_usage() {
    let frames = [
        text_frame("OK"),
        finish_frame("stop"),
        json!({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14}}),
    ];
    let with_done = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, true));
    let without_done = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, false));
    assert!(
        with_done.1.is_none() && without_done.1.is_none(),
        "{with_done:?} {without_done:?}"
    );
    assert_eq!(with_done.0, without_done.0);
    assert_eq!(
        without_done.0,
        vec![
            json!({"kind": "text_delta", "text": "OK"}),
            json!({"kind": "usage", "input_tokens": 13, "output_tokens": 1, "cached_input_tokens": null, "reasoning_tokens": null}),
            json!({"kind": "completed"}),
        ]
    );
}

/// A `length` finish then EOF keeps the truncation contract: the open tool
/// fragment is dropped and the turn settles Incomplete, never malformed.
#[test]
fn length_finish_without_done_sentinel_is_incomplete() {
    let frames = [
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "lookup", "arguments": "{\"city"}}]}}]}),
        finish_frame("length"),
    ];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, false));
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(events.last(), Some(&json!({"kind": "incomplete"})));
}

/// Without a finish reason nothing is synthesized: a stream that emitted
/// content and then closed is still a malformed ending, so a mid-answer
/// disconnect cannot be mistaken for a complete turn.
#[test]
fn eof_without_finish_reason_after_output_is_the_providers_cut() {
    // No finish reason and the connection closed on served text: the
    // provider cut the answer (gpt-5.6-luna, 33 of 34 terminal-less streams
    // post-commit, 2026-09-14). The served tokens are real, so the turn
    // settles Incomplete instead of a 502 nothing can fail over from; before
    // any output the stream stays terminal-less (see `cut_tests`).
    let frames = [text_frame("half an ans")];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, false));
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "half an ans"}),
            json!({"kind": "incomplete"}),
        ]
    );
}

/// The synthesized ending runs the same tool contract as `[DONE]`: a finish
/// other than `length` over a syntactically broken argument object stays
/// malformed rather than completing a corrupt call.
#[test]
fn eof_after_finish_keeps_the_strict_tool_argument_contract() {
    let frames = [
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "lookup", "arguments": "{\"city\": [1,}"}}]}}]}),
        finish_frame("tool_calls"),
    ];
    let (_, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &wire(&frames, false));
    let failure = failure.expect("broken arguments stay malformed");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}
#[test]
fn a_resellers_flat_error_frame_is_classified_by_its_reason_token() {
    // Novita declares failure inside a stream in the same flat envelope it
    // answers pre-stream (no `error` object, the token under `reason`); each
    // documented reason lands in its class with the sentence as detail.
    let cases = [
        (
            json!({"code": 429, "reason": "RATE_LIMIT_EXCEEDED", "message": "Too many requests, please try again later", "metadata": {}}),
            FailureClass::Throttled,
        ),
        (
            json!({"code": 429, "reason": "TOKEN_LIMIT_EXCEEDED", "message": "Token limit exceeded, please try again later", "metadata": {}}),
            FailureClass::Throttled,
        ),
        (
            json!({"code": 400, "reason": "INVALID_REQUEST_BODY", "message": "max_tokens must be less than or equal to 131072", "metadata": {}}),
            FailureClass::InvalidRequest,
        ),
        (
            json!({"code": 403, "reason": "NOT_ENOUGH_BALANCE", "message": "Insufficient balance", "metadata": {}}),
            FailureClass::ProviderQuota,
        ),
        (
            json!({"code": 401, "reason": "FAILED_TO_AUTH", "message": "failed to authenticate API key", "metadata": {}}),
            FailureClass::ProviderAuthentication,
        ),
        (
            json!({"code": 503, "reason": "SERVICE_NOT_AVAILABLE", "message": "Service unavailable", "metadata": {}}),
            FailureClass::ProviderInternal,
        ),
    ];
    for (payload, expected) in cases {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let events = normalizer
            .feed(&SseEvent {
                event: None,
                data: payload.to_string(),
            })
            .expect("a declared failure is an event, not a malformed frame");
        let [Event::Failed(failure)] = events.as_slice() else {
            panic!("expected one Failed event, got {events:?}");
        };
        assert_eq!(failure.failure_class, expected, "{payload}");
        let message = payload["message"].as_str().unwrap();
        assert!(
            failure
                .provider_detail
                .as_deref()
                .is_some_and(|detail| detail.contains(message)),
            "the provider's sentence rides the failure into the ledger: {failure:?}"
        );
    }
}

#[test]
fn a_frame_with_non_array_choices_names_its_shape_in_the_malformed_reason() {
    // The malformed reason carries the frame's key names (never its values)
    // so an unknown shape is diagnosable from the ledger.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": "nope", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).to_string(),
        })
        .expect_err("a non-array choices stays malformed");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    assert_eq!(
        failure.safe_message,
        "OpenAI-compatible choices must be an array (frame keys: choices, id, object, usage)"
    );
}

#[test]
fn a_frames_hostile_key_names_are_masked_in_the_malformed_reason() {
    // Keys are provider text: a newline, an ANSI escape, or a delimiter in a
    // key must not reach the ledger line; the key reads as `non-identifier`.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: json!({"choices": "nope", "id": "x", "line\nbreak": 1, "\u{1b}[31mred": 2, "a, b": 3}).to_string(),
        })
        .expect_err("a non-array choices stays malformed");
    assert_eq!(
        failure.safe_message,
        "OpenAI-compatible choices must be an array (frame keys: choices, id, non-identifier, non-identifier, non-identifier)"
    );
    assert!(!failure.safe_message.contains('\n'));
    assert!(!failure.safe_message.contains('\u{1b}'));
}

#[test]
fn an_in_stream_relay_decode_failure_relays_the_upstream_error() {
    // The same relay sentence declared inside a stream frame: the upstream
    // document's class and sentence decide.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let frame = json!({"error": {"code": 0, "type": "invalid_request_error", "message":
        "failed to decode error response: json: cannot unmarshal number into Go struct field \
         ResponseError.error.code of type string, raw: {\"error\":{\"code\":429,\"message\":\
         \"Rate limit exceeded, please retry later.\"}} trace_id: 92913336280c9c28f727ac9bfefbd89c"}});
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: frame.to_string(),
        })
        .expect("a declared failure is an event");
    let [Event::Failed(failure)] = events.as_slice() else {
        panic!("expected one Failed event, got {events:?}");
    };
    assert_eq!(failure.failure_class, FailureClass::Throttled);
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("Rate limit exceeded, please retry later.")));
}

#[test]
fn the_first_upstream_provider_label_is_kept_and_garbage_is_ignored() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let chunk = |provider: Value, content: &str| SseEvent {
        event: None,
        data: json!({
            "provider": provider,
            "choices": [{"index": 0, "delta": {"content": content}}]
        })
        .to_string(),
    };
    // Before any chunk names one there is nothing to settle.
    assert_eq!(normalizer.upstream_provider(), None);
    // An empty label, a non-string, and a non-printable label are not names.
    normalizer
        .feed(&chunk(json!(""), "a"))
        .expect("frame parses");
    normalizer
        .feed(&chunk(json!(7), "b"))
        .expect("frame parses");
    normalizer
        .feed(&chunk(json!("Az\u{7}ure"), "c"))
        .expect("frame parses");
    assert_eq!(normalizer.upstream_provider(), None);
    // The first real label wins and a later, different one never replaces it.
    normalizer
        .feed(&chunk(json!(" Azure "), "d"))
        .expect("frame parses");
    normalizer
        .feed(&chunk(json!("Amazon Bedrock"), "e"))
        .expect("frame parses");
    assert_eq!(normalizer.upstream_provider(), Some("Azure"));
    // A chunk without the field is the ordinary shape and changes nothing.
    normalizer
        .feed(&frame(json!({"index": 0, "delta": {"content": "f"}})))
        .expect("frame parses");
    assert_eq!(normalizer.upstream_provider(), Some("Azure"));
}

#[test]
fn an_over_long_upstream_provider_label_is_not_a_name() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let long = "x".repeat(crate::dialects::UPSTREAM_PROVIDER_MAX_CHARS + 1);
    normalizer
        .feed(&SseEvent {
            event: None,
            data: json!({"provider": long, "choices": [{"index": 0, "delta": {"content": "a"}}]})
                .to_string(),
        })
        .expect("frame parses");
    assert_eq!(normalizer.upstream_provider(), None);
}
