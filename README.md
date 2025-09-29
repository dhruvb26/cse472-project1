## Introduction

The purpose of this project was to explore **social media data crawling**, experiment with **large language models (LLMs)**, practice **prompt engineering**, and conduct **exploratory data analysis (EDA)**. For this, I chose the open-source social media platform [Mastodon](https://mastodon.social), which offers a publicly available API.

This report outlines my thought process, the steps I followed, and the reasoning behind each decision. The entire code for this project can be found on [GitHub](https://github.com/dhruvb26/cse472-project1).

## Setup

### Prerequisites

- Python 3.13 or higher
- [astral-uv](https://astral.sh/uv/)

### Environment Setup

1. Clone this repository and navigate to the project directory

2. Create and configure your environment file:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and fill in your credentials from Mastodon.

3. Install dependencies using uv:
   ```bash
   uv sync
   ```

### Project Structure

The project is organized into several independent modules, with common utilities shared across them:

- **Core Utilities**

  - [helpers.py](https://github.com/dhruvb26/cse472-project1/blob/main/helpers.py) - Common functions and utilities used across modules

- **Data Collection & Analysis**

  - [posts.py](https://github.com/dhruvb26/cse472-project1/blob/main/posts.py) - Collects and analyzes posts by hashtags/keywords, generates post network graphs
  - [users.py](https://github.com/dhruvb26/cse472-project1/blob/main/users.py) - Manages user data collection and user network graph generation
  - [network_measures.py](https://github.com/dhruvb26/cse472-project1/blob/main/network_measures.py) - Computes and analyzes network metrics for user interactions
  - [generate_wordcloud.py](https://github.com/dhruvb26/cse472-project1/blob/main/generate_wordcloud.py) - Creates visual word clouds from post content

- **Analysis Notebooks**
  - [content_analysis_notebook.ipynb](https://github.com/dhruvb26/cse472-project1/blob/main/content_analysis_notebook.ipynb) - Detailed content analysis and visualization
  - [content_analysis_notebook.pdf](https://github.com/dhruvb26/cse472-project1/blob/main/content_analysis_notebook.pdf) - PDF export with complete notebook output

Each Python module can be run independently based on your analysis needs.

## Data Collection

The first step was **data acquisition** from Mastodon. Since the platform provides an API, I used the Python library [Mastodon.py](https://mastodonpy.readthedocs.io/en/stable/) to handle the crawling process.

I collected **two datasets**:

1. **Posts (toots)**
2. **Users**

These datasets allowed me to later build a network and uncover relationships between users and their activity.

### Keyword-Based Data (Posts)

For collecting posts, I relied on the **hashtags/keywords** provided in the project description.

- I implemented a `find_posts` function in [posts.py](https://github.com/dhruvb26/cse472-project1/blob/main/posts.py) with two key parameters:

  - `hashtag_limit` → controlled the distribution of keywords across posts.
  - `post_limit` → controlled the total number of posts collected.

- Starting with ~10 initial keywords, I evenly distributed the number of posts across them.
- To ensure the resulting network was not too sparse, I **ignored posts with fewer than 3 replies**.
- For each collected post, I also retrieved its replies and stored them in the same structure as the original toot.
- I applied **deduplication** across all posts to avoid redundancy.

This process resulted in **~600 posts**, saved in [posts.json](https://github.com/dhruvb26/cse472-project1/blob/main/posts.json). The file is a top-level array where each element represents a toot.

### User-Based Data

In addition to posts, I collected user data by performing a **Breadth-First Search (BFS)** starting from a set of **seed users**:

- I expanded along both **followers** and **following** lists, with a **limit of 10 per user** to avoid overly large datasets.
- The structure of [users.json](https://github.com/dhruvb26/cse472-project1/blob/main/users.json) is as follows:
  - Each element contains a `user` key.
  - Optional `followers` and `following` arrays are included if these lists are public.
  - If the lists are private, only the `user` key is saved.

This design choice was necessary because Mastodon’s API does not directly reveal identifiers for a user’s followers or followings.

### Supporting Data

To simplify data loading, I also saved the **seed users** and **keywords** used in a separate [data.json](https://github.com/dhruvb26/cse472-project1/blob/main/data.json) file. This file is read when executing the crawling logic.

## Network Construction and Visualization

For constructing the networks from collected data above I used the [`networkx`](https://networkx.org/) library in Python to help me in this task and [`Gephi`](https://gephi.org/) to visualize the final result.

### Information Diffusion Network

This network visualizes the interaction patterns between posts (toots) in the collected dataset. Each node represents a toot, with directed edges indicating reply relationships between posts. An edge from post A to post B means that post B is a reply to post A, as determined by the `in_reply_to` field.

Node Properties:

- **Color**: Based on the post's primary hashtag from its `tags` array
  - Posts with the same primary hashtag form color-coded communities
  - Posts without tags are colored gray

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/posts_network.png" alt="Information Diffusion Network" width="500">

### Friendship Network

This network represents the social connections between Mastodon users, constructed from the user-based data collection. Each node represents a unique user account, with undirected edges indicating follower/followee relationships between users. An edge between users A and B means that either A follows B, or B follows A, or both.

Node Properties:

- **Size**: Scaled between 5-50 pixels, proportional to the user's total degree (normalized)
- **Color**: Base color with intensity variations:
  - Higher degree nodes appear darker (70% brightness)
  - Lower degree nodes appear lighter (130% brightness)
  - This creates a natural visual hierarchy where more connected users stand out

This visualization helps identify:

- Social hubs (large, dark nodes)
- Peripheral users (small, light nodes)
- Connection patterns and community structures

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_network.png" alt="Friendship Network" width="500">
 
 ## Network Measures

To analyze the structural properties of the friendship network, I computed several network measures using NetworkX. These analyses reveal key insights about user behavior, network topology, and information flow patterns.

### Degree Distribution

I analyzed the degree distribution in two ways to understand the network's connectivity patterns:

#### Raw Distribution

   <img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_degree_dist.png" alt="Degree Distribution" width="500">

This histogram reveals several key insights about user connectivity:

- Most users have relatively few connections (1-5 friends)
- There's a long tail of users with higher degrees, indicating the presence of "super-connectors"
- The average degree is significantly lower than the maximum degree, suggesting a hierarchical social structure
- The distribution shape aligns with typical social network patterns where most users maintain a small, manageable number of connections

#### Probability Distribution P(k)

   <img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_degree_dist_P(k).png" alt="Degree Distribution P(k)" width="500">

The log-log plot of degree probability reveals:

- A rough power-law relationship (approximately straight line in log-log scale)
- This indicates a scale-free network structure, common in social networks
- The slope suggests that the probability of finding highly-connected users decreases exponentially
- The network follows similar patterns to other social platforms like Facebook, where a small number of users act as major hubs

### Centrality Measures

To understand how users influence information flow in the network, I analyzed two key centrality measures and their relationship with node degrees.

#### Betweenness Centrality vs Degree

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_degree_vs_betweenness.png" alt="Degree vs Betweenness" width="500">

The analysis reveals a strong positive correlation (0.610) between betweenness centrality and node degree, indicating that users with more connections generally serve as important bridges in the network. This relationship suggests that high-degree users often facilitate information flow between different parts of the network. However, the relationship isn't perfect – several medium-degree users show surprisingly high betweenness centrality, suggesting they play crucial roles in connecting otherwise disparate communities. These users might be bridging different interest groups or social circles despite not having an exceptionally large number of connections themselves.

The scatter pattern also shows significant variation, particularly among high-degree users. Some users with similar degrees have vastly different betweenness centrality values, indicating that not all well-connected users are equally important for network-wide information flow. This suggests that the position of connections in the network matters as much as the number of connections.

#### Closeness Centrality vs Degree

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_degree_vs_closeness.png" alt="Degree vs Closeness" width="500">

The relationship between closeness centrality and degree shows an even stronger positive correlation (0.644), but with interesting nuances. This correlation suggests that users with more connections tend to be more centrally positioned in the network, allowing them to reach other users through shorter paths. The stronger correlation compared to betweenness centrality (0.644 vs 0.610) indicates that having more connections is more reliably associated with being centrally positioned than with being a bridge between communities.

However, the substantial variance in the relationship reveals important exceptions. Some users with relatively few connections achieve high closeness centrality, suggesting they've formed strategic connections with well-positioned users. This pattern indicates a network where information can potentially spread efficiently even through users with fewer connections, as long as those connections are well-placed. The distribution also suggests that while high-degree users generally have good closeness centrality, there's a diminishing returns effect where additional connections don't necessarily lead to proportionally better positioning in the network.

### Local Network Structure

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/users_local_friends_dist.png" alt="Local Friends Distribution" width="500">

The distribution of local friend connections provides insights into social clustering:

- The mean local friend count suggests moderate local clustering
- A right-skewed distribution indicates most users belong to smaller, tighter communities
- The standard deviation shows considerable variation in local social circle sizes
- Some users bridge between different-sized communities, as shown by the spread
- The shape suggests users tend to connect within their "social class" (degree assortativity)

Key findings from local structure:

- Users generally form small, interconnected groups
- Some users act as bridges between different-sized communities
- The network shows signs of homophily, where users connect to others with similar connectivity levels
- Local clustering suggests information spreads efficiently within communities but may take longer to reach the broader network

### Global Network Structure

The global network analysis provides insights into the overall connectivity patterns of the entire network:

- **Average Degree**: 2.48 connections per user
- **Total Nodes**: 2,105 users
- **Network Density**: This relatively low average degree suggests a sparse network, typical of large social networks where users maintain connections with a small fraction of the total user base

This global measure was calculated by dividing the total degree of the network by the number of nodes, providing a high-level view of network connectivity. The sparsity is expected in social networks, as users typically cannot maintain connections with a large portion of the network.

For detailed implementation of these network measures and to generate updated statistics, you can run [network_measures.py](https://github.com/dhruvb26/cse472-project1/blob/main/network_measures.py).

## Content Analysis

To understand the main topics and themes discussed in the collected posts, I performed content analysis using the Unsloth `unsloth/llama-3-8b-bnb-4bit` language model. This analysis involved keyword extraction from approximately 600 posts and visualization of the prominent themes.

Here is the link to the [Colab Notebook](https://colab.research.google.com/drive/17itXDoBFpcPKa6Vj0fjCgjrN3qvUtLoO?usp=sharing).

### Prompt Engineering

I crafted a specialized prompt to extract meaningful keywords while maintaining consistency and quality:

```
prompt_template = """
System:
You are an information-extraction engine. Return valid JSON only.

Rules:
Extract up to 3 keywords that best represent the main topics of the post.
- Be concise, lowercase, no hashtags, no emojis or punctuation except hyphens.
- Output a maximum of 3 keywords.
- Return valid JSON only, matching the schema.

Schema (JSON):
{{
 "post_id": "<string>",
 "keywords": [{{"text":"<string>","confidence": <float>}}]
}}

Example:
Input post_id: "p_42"
Input text: "We fine-tuned a multilingual sentence transformer using hard negatives from click logs."
Output:
{{"post_id":"p_42","keywords":[
 {{"text":"sentence transformer","confidence":0.92}},
 {{"text":"hard negatives","confidence":0.84}},
 {{"text":"click logs","confidence":0.80}}
]}}

Input:
post_id: {post_id}
text: {post_text}

Output:
"""
```

The prompt was designed with several key features:

- Structured JSON output for easy processing
- Confidence scores for each keyword
- Support for multi-word concepts (e.g., "sentence transformer")
- Consistent formatting rules (lowercase, no special characters)

### Keyword Extraction Results

The model processed approximately 600 posts in 52 minutes, generating keywords with confidence scores. Here are some representative examples:

1. AI-focused discussion:

```json
{
  "post_id": "115228947625944347",
  "keywords": [
    { "text": "ai", "confidence": 0.92 },
    { "text": "chatgpt", "confidence": 0.84 },
    { "text": "llm", "confidence": 0.8 }
  ],
  "is_reply": false
}
```

2. Technical discussion:

```json
{
  "post_id": "115079614293402819",
  "keywords": [
    { "text": "unsupervised learning", "confidence": 0.92 },
    { "text": "anomaly detection", "confidence": 0.84 },
    { "text": "streaming", "confidence": 0.8 }
  ],
  "is_reply": true
}
```

### Theme Analysis

<img src="https://github.com/dhruvb26/cse472-project1/raw/main/ai_topics_wordcloud.png" alt="Word Cloud" width="500">

The word cloud visualization reveals several prominent themes and patterns:

1. **Core AI Technologies**:

   - Frequent discussion of AI, machine learning, and neural networks
   - Strong presence of LLM-related terms (ChatGPT, GPT, language models)
   - Emphasis on practical applications and tools

2. **Technical Concepts**:

   - Data science and analytics terminology
   - Programming and development topics
   - Research and academic discussions

3. **Emerging Patterns**:
   - High engagement around AI ethics and safety
   - Significant interest in open-source AI projects
   - Discussions about AI's impact on various fields

### Unexpected Findings

Several interesting patterns emerged from the analysis:

1. **Community Focus**:

   - Strong emphasis on open-source alternatives to commercial AI
   - Frequent discussions about AI safety and ethics
   - High engagement in educational content

2. **Technical Depth**:
   - Detailed technical discussions rather than surface-level AI news
   - Strong focus on implementation details and practical applications
   - Active sharing of research papers and technical resources

The results are stored in [results.jsonl](https://github.com/dhruvb26/cse472-project1/blob/main/results.jsonl) for further analysis or verification.
