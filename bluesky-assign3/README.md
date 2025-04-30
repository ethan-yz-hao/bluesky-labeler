# Bluesky labeler starter code

You'll find the starter code for Assignment 3 in this repository. More detailed
instructions can be found in the assignment spec.

## The Python ATProto SDK

To build your labeler, you'll be using the AT Protocol SDK, which is documented [here](https://atproto.blue/en/latest/).

## Automated labeler

The bulk of your Part I implementation will be in `automated_labeler.py`. You are
welcome to modify this implementation as you wish. However, you **must**
preserve the signatures of the `__init__` and `moderate_post` functions,
otherwise the testing/grading script will not work. You may also use the
functions defined in `label.py`. You can import them like so:

```
from .label import post_from_url
```

For Part II, you will create a file called `policy_proposal_labeler.py` for your
implementation. You are welcome to create additional files as you see fit.

## Input files

For Part I, your labeler will have as input lists of T&S words/domains, news
domains, and a list of dog pictures. These inputs can be found in the
`labeler-inputs` directory. For testing, we have CSV files where the rows
consist of URLs paired with the expected labeler output. These can be found
under the `test-data` directory.

## Testing

We provide a testing harness in `test-labeler.py`. To test your labeler on the
input posts for dog pictures, you can run the following command and expect to
see the following output:

```
% python test_labeler.py labeler-inputs test-data/input-posts-dogs.csv
The labeler produced 20 correct labels assignments out of 20
Overall ratio of correct label assignments 1.0
```

## Part II: Hate Speech Detection

For Part II, we've implemented a sophisticated hate speech detection system that can identify harmful content in Bluesky posts. The implementation uses a multi-stage approach combining lexicon-based filtering with advanced AI analysis.

### Key Features

-   **Multi-stage Detection Pipeline**: Uses HurtLex lexicon for initial screening followed by OpenAI's models for deeper content analysis
-   **Multimodal Analysis**: Analyzes both text and images in posts for comprehensive detection
-   **Detailed Explanations**: Provides reasoning behind classifications to improve transparency
-   **Configurable Thresholds**: Adjustable sensitivity for different use cases

### Requirements

To run the hate speech detection system, you'll need to install the following dependencies:

```bash
pip install torch atproto python-dotenv pandas openai requests spacy
python -m spacy download en_core_web_sm
```

### Configuration

The system requires the following environment variables in a `.env` file:

```
USERNAME=your_bluesky_username
PW=your_bluesky_password
OPENAI_API_KEY=your_openai_api_key

```

### Usage

You can analyze individual posts for hate speech content:

```bash
python -m pylabel.policy_proposal_labeler --url https://bsky.app/profile/user.bsky.social/post/postid --threshold 0.5 --model gpt-4o
```

Or test the system on a dataset of posts:

```bash
python test_policy_proposal_labeler.py test-data/bluesky_hate_speech_sample.csv results.csv --model gpt-4o --threshold 0.5
```

### How It Works

1. **Lexicon-Based Filtering**: Uses HurtLex, a multilingual hate speech lexicon, to perform initial screening
2. **NLP Processing**: Applies spaCy for text normalization and lemmatization
3. **Category-Weighted Analysis**: Assigns different weights to hate speech categories based on severity
4. **AI-Powered Analysis**: For complex cases, uses OpenAI's models to analyze context and nuance
5. **Image Analysis**: Processes images in posts to detect visual hate speech content

### Output Format

The system produces detailed analysis results including:

-   Hate speech probability score (0.0-1.0)
-   Binary classification (hate speech or not)
-   Explanation of the classification decision
-   Detection method used (hurtlex, openai, or combined)
-   Matched hate speech terms and categories (if any)

This implementation provides a robust foundation for content moderation on social platforms, balancing efficiency with accuracy through its multi-stage approach.
