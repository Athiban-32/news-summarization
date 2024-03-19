import wordcloud
import matplotlib.pyplot as plt

# Define extracted text from each topic (replace placeholders with actual text)
topics = {
    "Topic 0": ["album", "date", "headline", ...],  # Replace with words from topic 0
    "Topic 1": ["fan", "play", "photo", ...],  # Replace with words from topic 1
    # ... add other topics
}

# Set common options for all word clouds
wordcloud_options = {
    "width": 400,
    "height": 300,
    "background_color": "white",
    "max_words": 100,  # Adjust as needed
    "colormap": "Set2",  # Adjust colormap as desired
}

# Create word cloud for each topic
for topic_name, words in topics.items():
    wordcloud = wordcloud.WordCloud(**wordcloud_options).generate(" ".join(words))

    plt.figure(figsize=(6, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(topic_name)
    plt.show()
