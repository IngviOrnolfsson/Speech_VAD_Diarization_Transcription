"""CLI helper for the conversation VAD labeler package."""

from __future__ import annotations

from pathlib import Path

from conversation_vad_labeler.conversation import process_conversation


def _default_example_inputs() -> dict[str, str]:
    base = Path("examples/recordings")
    return {
        "P1": str(base / "EXP9_None_p1_trial2.wav"),
        "P2": str(base / "EXP9_None_p2_trial2.wav"),
    }


def main() -> None:
    print("Running example conversation processing...")
    speakers_audio = _default_example_inputs()
    output_directory = "outputs/test"

    process_conversation(
        speakers_audio=speakers_audio,
        output_dir=output_directory,
        whisper_device="auto",
        interactive_energy_filter=False,
        batch_size=240.0,#sec
    )

    print("\nExample completed. Check the output files in the generated directory.")


if __name__ == "__main__":
    main()