# Social Media Data Analysis

This repository contains the source code for the Social Media Data Analysis project done as part of the CSE472: Social Media Mining course at the Arizona State University (Fall 2025).

## Project Description

The aim for this project was to gain expertise in social media data crawling, utilizing large language models
(LLMs), learning how to create effective prompts, and conducting exploratory analysis of the extracted
data.

For this project I used the Mastodon social media platform to crawl data through the [Mastodon.py](https://mastodonpy.readthedocs.io/en/stable/) library.

## Setup

I used [astral-sh/uv](https://docs.astral.sh/uv/) to manage the dependencies and the virtual environment for the project. To install the dependencies and activate the virtual environment, run the following command:

```bash
uv sync

source .venv/bin/activate
```

## Usage

This section will cover the usage of different files and their purpose to complete every step of the project. Each file is responsible for a specific part of the project and uses the [helpers.py](helpers.py) file to handle common functions and the [data.json](data.json) file to store any data that is needed.

### Step 1: Obtaining Mastodon API Credentials

The `_auth` function is used to authenticate with the Mastodon API and the `get_client` function is used to get the client object. Both of these functions are used in the [users.py](users.py) and [posts.py](posts.py) files.

### Step 2: Data Collection

Keyword-Based Data Collection: I saved the given keywords in the [data.json](data.json) file and used the `find_posts` function in [posts.py](posts.py) to collect posts based on the keywords. The file is responsible for finding the posts and also for any subsequent processing on this particular data.

The file's functionality is controlled through cli arguments. To run the file with the default arguments, run the following command:

```bash
uv run posts.py # Creates a network graph of already discovered posts in posts_network.gexf
```

If you want to discover new posts, you can run the following command:

```bash
uv run posts.py --mode find # Saves the new posts to posts.json
```

Check out the [posts.json](posts.json) for ~600 discovered posts.
