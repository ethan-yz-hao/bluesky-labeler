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

## Part II: Job Scam Detection

For Part II, we've implemented a job scam detection system that can identify fraudulent money-making schemes and employment scams on Bluesky. The implementation uses pattern recognition and heuristic analysis to flag suspicious content.

### Key Features

-   **Pattern-Based Detection**: Uses regex patterns to identify common job scam indicators
-   **Multi-factor Analysis**: Considers earnings claims, barrier-to-entry promises, and suspicious calls-to-action
-   **Transparent Explanations**: Provides reasoning behind classifications for better understanding
-   **Configurable Threshold**: Adjustable sensitivity to balance precision and recall

### Requirements

To run the job scam detection system, you'll need to install the following dependencies:

```bash
pip install atproto python-dotenv pandas scikit-learn
```

### Configuration

The system requires the following environment variables in a `.env` file:

```
USERNAME=your_username
PW=your_password
```

### Usage

You can analyze individual posts for job scam content:

```bash
python -m pylabel.policy_proposal_labeler --url https://bsky.app/profile/user.bsky.social/post/postid --threshold 0.5
```

Or test the system on a dataset of posts:

```bash
python test_policy_proposal_labeler.py labeled_money_posts.csv results.csv --threshold 0.5
```

### How It Works

1. **Keyword Matching**: Identifies job-related content using common employment terminology
2. **Scam Pattern Detection**: Applies regex patterns to detect:
    - Unrealistic earnings claims ("earn $500/day")
    - Zero-barrier promises ("no experience needed")
    - External platform funneling (Telegram/WhatsApp links)
3. **Call-to-Action Analysis**: Identifies suspicious CTAs that pressure users to take immediate action
4. **Probability Scoring**: Combines multiple signals to generate a scam probability score
5. **Threshold-Based Classification**: Uses configurable threshold to make final determination

### Data Collection

The test dataset was created using:

-   A custom Bluesky hashtag scraper targeting money-making related tags
-   Manual labeling supplemented by GPT-4 analysis for ground truth
-   Diverse post collection across multiple hashtags and content types

### Output Format

The system produces detailed analysis results including:

-   Job scam probability score (0.0-1.0)
-   Binary classification (scam or not)
-   Explanation of the classification decision
-   Detection method used
-   Matched scam patterns (if any)

This implementation provides an effective tool for identifying potentially harmful job scams on social platforms, helping protect users from financial fraud and exploitation.
