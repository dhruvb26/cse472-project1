import json
import logging
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

logger = logging.getLogger(__name__)


def read_keywords_from_jsonl(file_path):
    """
    Read keywords from JSONL file and return a list of all keywords with their confidence scores.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        list: List of tuples (keyword_text, confidence_score)
    """
    keywords = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    post_data = json.loads(line)
                    post_keywords = post_data.get("keywords", [])

                    # Extract keywords with their confidence scores
                    for keyword_obj in post_keywords:
                        if isinstance(keyword_obj, dict) and "text" in keyword_obj:
                            text = keyword_obj["text"].lower().strip()
                            confidence = keyword_obj.get("confidence", 1.0)

                            # Filter out very short keywords (likely noise)
                            if len(text) > 1:
                                keywords.append((text, confidence))

                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Could not parse line {line_num}: {e}")
                    raise

    except FileNotFoundError:
        logger.error(f"Error: File {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error in read_keywords_from_jsonl: {e}")
        raise

    return keywords


def calculate_weighted_frequencies(keywords, confidence_weight=0.5):
    """
    Calculate keyword frequencies with optional confidence weighting.

    Args:
        keywords (list): List of (keyword_text, confidence_score) tuples
        confidence_weight (float): Weight for confidence scores (0-1)

    Returns:
        dict: Dictionary of keyword frequencies
    """
    frequency_counter = Counter()

    for keyword_text, confidence in keywords:
        weight = 1 + (confidence * confidence_weight)
        frequency_counter[keyword_text] += weight

    return dict(frequency_counter)


def filter_stop_words(frequencies):
    """
    Filter out common stop words and very common terms that don't add value.

    Args:
        frequencies (dict): Keyword frequency dictionary
        custom_stop_words (set): Additional stop words to filter

    Returns:
        dict: Filtered frequency dictionary
    """
    with open("data.json", "r") as f:
        data = json.load(f)
        stop_words = set(data["stopwords"])

    filtered_frequencies = {
        keyword: freq
        for keyword, freq in frequencies.items()
        if keyword.lower() not in stop_words and len(keyword) > 1
    }

    return filtered_frequencies


def create_wordcloud(
    frequencies, output_path="ai_topics_wordcloud.png", width=2400, height=1600
):
    """
    Create and save a high-quality word cloud visualization.

    Args:
        frequencies (dict): Keyword frequency dictionary
        output_path (str): Path to save the word cloud image
        width (int): Width of the word cloud
        height (int): Height of the word cloud
    """

    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=300,
        relative_scaling=0.6,  # type: ignore
        colormap="OrRd",
        max_font_size=150,
        min_font_size=15,
        prefer_horizontal=0.7,
        margin=15,
        random_state=42,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(24, 16))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(
        output_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none"
    )


def print_top_keywords(frequencies, top_n=20):
    """
    Print the top N keywords by frequency.

    Args:
        frequencies (dict): Keyword frequency dictionary
        top_n (int): Number of top keywords to display
    """
    sorted_keywords = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} Keywords by Frequency:")
    print("-" * 40)
    for i, (keyword, freq) in enumerate(sorted_keywords[:top_n], 1):
        print(f"{i:2d}. {keyword:<20} ({freq:.2f})")


def main():
    """Main function to generate word cloud from JSONL data."""

    keywords = read_keywords_from_jsonl("results.jsonl")
    frequencies = calculate_weighted_frequencies(keywords, confidence_weight=0.3)
    filtered_frequencies = filter_stop_words(frequencies)

    print_top_keywords(filtered_frequencies)
    create_wordcloud(filtered_frequencies)


if __name__ == "__main__":
    main()
