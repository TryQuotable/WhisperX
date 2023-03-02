import numpy as np
import pandas as pd


def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):
    words = []
    for seg in result_segments:
        wdf = seg['word-segments']

        for wdx, wrow in wdf.iterrows():
            word = seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
            words.append((word, wrow))

    sequences = []
    current_sequence = []
    for word, wrow in words:
        if len(current_sequence) > 0 and (current_sequence[-1][0][-1] in [".", "!", "?"]):
            sequences.append(current_sequence)
            current_sequence = []
        current_sequence.append((word, wrow))

    sequences.append(current_sequence)

    sdf_dict = []
    for sequence in sequences:
        rows = pd.DataFrame([r for _, r in sequence])
        sdf_dict.append(
            {
                "start": rows["start"].min(),
                "end": rows["end"].max(),
                "num_words": len(rows),
                "words": sequence
            },
        )
    sdf = pd.DataFrame(sdf_dict, columns=["start", "end", "num_words", "words"])

    speakers = []
    for wdx, wrow in sdf.iterrows():
        if not np.isnan(wrow['start']):
            diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
            diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
            # remove no hit
            if not fill_nearest:
                dia_tmp = diarize_df[diarize_df['intersection'] > 0]
            else:
                dia_tmp = diarize_df
            if len(dia_tmp) == 0:
                speaker = None
            else:
                speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
        else:
            speaker = None

        for i in range(int(wrow['num_words'])):
            speakers.append((speaker, wrow, wrow['words'][i]))

    # create word level segments for .srt
    word_seg = []
    wseg = pd.DataFrame(speakers)
    for speaker, _, (word, wrow) in speakers:
        if wrow["start"] is not None:
            if speaker is None or speaker == np.nan:
                speaker = "UNKNOWN"
            word_seg.append(
                {
                    "start": wrow["start"],
                    "end": wrow["end"],
                    "text": f"[{speaker}]: " + word
                }
            )

    return result_segments, word_seg

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
