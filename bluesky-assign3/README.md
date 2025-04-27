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

For Part II, we implement a hate speech detection system that can identify profiles that frequently post harmful content. This implementation uses a pre-trained model from Hugging Face to analyze post content.

### Requirements

To run the hate speech detection scripts, you'll need to install the following dependencies:

```bash
pip install torch transformers atproto python-dotenv pandas
```

### Scripts

The implementation consists of two main scripts:

1. **test_hate_speech_model.py**: A utility script to test the hate speech detection model locally.
2. **profile_hate_speech_analyzer.py**: A script that analyzes Bluesky profiles for hate speech content.
3. **policy_proposal_labeler.py**: The main implementation for Part II that labels profiles based on hate speech detection.

### Testing the Hate Speech Model

You can test the hate speech detection model with sample texts or your own data:

```bash
# Test with built-in sample texts
python test_hate_speech_model.py --sample-text

# Test with a JSON file containing posts
python test_hate_speech_model.py --input posts.json --text-field text

# Test with a CSV file
python test_hate_speech_model.py --input posts.csv --text-field text
```

### Analyzing Bluesky Profiles

To analyze a Bluesky profile for hate speech content, run:

```bash
# Analyze specific handles
python profile_hate_speech_analyzer.py --handles user1.bsky.social user2.bsky.social

# Analyze handles from a file
python profile_hate_speech_analyzer.py --handles-file handles.txt

# Customize analysis parameters
python profile_hate_speech_analyzer.py --handles user.bsky.social --post-limit 100 --threshold 0.7
```

### Model Information

The hate speech detection system uses the [dehatebert-mono-english](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english) model from Hugging Face, which is fine-tuned for detecting hate speech in English text.

### Output Format

The profile analyzer outputs a JSON file with detailed analysis results, including:

-   Number of posts analyzed per profile
-   Count and ratio of hate speech posts
-   Detailed information about posts classified as hate speech
-   Overall summary statistics

This data can be used to identify profiles that frequently post harmful content and may require moderation.
