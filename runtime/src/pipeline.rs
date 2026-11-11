//! Two capacity-one batch queues; workers own their codecs.
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;

type StageResult<T> = Result<T, String>;

/// Assemble directly from a fallible frame source; EOF alone releases a partial tail.
/// Errors are never converted to EOF. A zero batch size fails before reading.
pub fn next_batch<T>(
    size: usize,
    mut next: impl FnMut() -> Result<Option<T>, String>,
) -> Result<Option<Vec<T>>, String> {
    if size == 0 {
        return Err("batch size must be at least 1".to_owned().into());
    }
    let mut batch = Vec::with_capacity(size);
    while batch.len() < size {
        match next()? {
            Some(frame) => batch.push(frame),
            None => break,
        }
    }
    Ok((!batch.is_empty()).then_some(batch))
}

/// Preserve sub-integer FPS from video-rs' f32 API to micro-fps precision.
pub fn frame_rate_parts(fps: f32) -> Result<(i32, i32), String> {
    let scaled = (f64::from(fps) * 1_000_000.0).round();
    if !fps.is_finite()
        || fps <= 0.0
        || scaled < 1.0
        || scaled > i32::MAX as f64
    {
        return Err(format!(
            "invalid or unsupported source frame rate: {fps}"
        ));
    }
    let mut a = scaled as i32;
    let mut b = 1_000_000;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Ok((scaled as i32 / a, 1_000_000 / a))
}

fn joined<T>(
    result: thread::Result<StageResult<T>>,
    stage: &str,
) -> StageResult<T> {
    result.unwrap_or_else(|_| Err(format!("{stage} panicked")))
}

pub fn run<I: Send, O: Send, R: Send>(
    produce: impl FnOnce(SyncSender<I>) -> StageResult<()> + Send,
    mut transform: impl FnMut(I) -> StageResult<O>,
    consume: impl FnOnce(Receiver<O>) -> StageResult<R> + Send,
) -> StageResult<R> {
    thread::scope(|scope| {
        let (input_tx, input_rx) = sync_channel(1);
        let (output_tx, output_rx) = sync_channel(1);
        let producer = scope.spawn(move || produce(input_tx));
        let consumer = scope.spawn(move || consume(output_rx));
        // Catch unwind so both channels are disconnected BEFORE joining workers.
        let middle =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for batch in input_rx.iter() {
                    let output = transform(batch)?;
                    output_tx
                        .send(output)
                        .map_err(|_| "encoder disconnected".to_owned())?;
                }
                Ok(())
            }));
        drop(input_rx);
        drop(output_tx);
        let produced = joined(producer.join(), "decoder");
        let consumed = joined(consumer.join(), "encoder");
        let transformed = joined(middle, "inference");
        // Retain root errors even when another stage reports disconnection.
        let mut errors = Vec::new();
        if let Err(error) = produced {
            errors.push(error);
        }
        if let Err(error) = transformed {
            errors.push(error);
        }
        match consumed {
            Ok(result) if errors.is_empty() => Ok(result),
            Ok(_) => Err(errors.join("; ")),
            Err(error) => {
                errors.push(error);
                Err(errors.join("; "))
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_batch_size() {
        assert!(next_batch::<usize>(0, || panic!("must not read")).is_err());
    }

    #[test]
    fn validates_source_frame_rate() {
        assert_eq!(frame_rate_parts(25.0).unwrap(), (25, 1));
        for fps in [
            0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ] {
            assert!(frame_rate_parts(fps).is_err());
        }
    }
}
