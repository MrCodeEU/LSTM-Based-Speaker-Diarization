import pyannote.metrics.diarization as dia
from pyannote.core import Annotation
from pyannote.core.utils.types import Label


def eval_diarization(predicted_segments: Annotation, ground_truth_segments: Annotation, collar=0.25,
                     calculate_eer=True):
    """Evaluates speaker diarization performance."""

    # Calculate DER (using pyannote.metrics)
    der = dia.DiarizationErrorRate(collar=collar, skip_overlap=True)
    der_result = der(ground_truth_segments, predicted_segments)
    print(f"Diarization Error Rate (DER): {der_result:.2%}")

    jer = dia.JaccardErrorRate(collar=collar, skip_overlap=True)
    result = jer(ground_truth_segments, predicted_segments)
    print(f"JER: {abs(result)}")

    purity = dia.DiarizationPurity(collar=collar, skip_overlap=True)
    purity_result = purity(ground_truth_segments, predicted_segments)
    print(f"Purity: {purity_result:.2%}")

    coverage = dia.DiarizationCoverage(collar=collar, skip_overlap=True)
    coverage_result = coverage(ground_truth_segments, predicted_segments)
    print(f"Coverage: {coverage_result:.2%}")

    completeness = dia.DiarizationCompleteness(collar=collar, skip_overlap=True)
    completeness_result = completeness(ground_truth_segments, predicted_segments)
    print(f"Completeness: {completeness_result:.2%}")

    ier = dia.IdentificationErrorRate(collar=collar, skip_overlap=True)
    ier_result = ier(ground_truth_segments, predicted_segments)
    print(f"Identification Error Rate (IER): {ier_result:.2%}")

    return der_result


def annotation_to_rttm(annotation: Annotation, file_name: str, output_path: str = None):
    """Converts a PyAnnote annotation object into an RTTM file.

    Args:
        annotation: The PyAnnote annotation object.
        file_name: The name of the audio file corresponding to the annotation.
        output_path: (Optional) The path to write the RTTM file. If not provided,
            the function will return the RTTM content as a string.
    """

    # Make sure the annotation timeline is not empty
    if annotation.get_timeline().empty():
        raise ValueError("Annotation timeline is empty.")

    rttm_lines = []
    for segment, track, label in annotation.itertracks(yield_label=True):
        start_time = segment.start
        duration = segment.duration
        speaker_label = label if isinstance(label, Label) else f"SPEAKER_{track}"

        rttm_line = (
            f"SPEAKER {file_name} 1 "
            f"{start_time:.3f} {duration:.3f} <NA> <NA> "
            f"{speaker_label} <NA> <NA>\n"
        )
        rttm_lines.append(rttm_line)

    rttm_content = "".join(rttm_lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(rttm_content)
    else:
        return rttm_content
