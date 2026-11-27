"""Unit tests for EmotionDetection.emotion_detection.

The Watson NLP endpoint is mocked, so these tests run offline.
Set RUN_LIVE_TESTS=1 to also run the tests against the real Watson service
(only reachable from the Skills Network lab environment).
"""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from EmotionDetection.emotion_detection import emotion_detector

URL = ('https://sn-watson-emotion.labs.skills.network/v1/'
       'watson.runtime.nlp.v1/NlpService/EmotionPredict')
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EXPECTED_KEYS = {'anger', 'disgust', 'fear', 'joy', 'sadness', 'dominant_emotion'}


def make_response(anger=0.0, disgust=0.0, fear=0.0, joy=0.0, sadness=0.0):
    """Build a fake requests.Response shaped like the Watson payload."""
    payload = {
        "emotionPredictions": [{
            "emotion": {
                "anger": anger,
                "disgust": disgust,
                "fear": fear,
                "joy": joy,
                "sadness": sadness,
            }
        }]
    }
    response = MagicMock()
    response.status_code = 200
    response.text = json.dumps(payload)
    return response


@patch('EmotionDetection.emotion_detection.requests.post')
class TestEmotionDetector(unittest.TestCase):
    """Tests emotion_detector with the HTTP call mocked."""

    def test_sends_expected_request(self, mock_post):
        mock_post.return_value = make_response(joy=0.9)

        emotion_detector("I love this new technology.")

        mock_post.assert_called_once_with(
            URL,
            json={"raw_document": {"text": "I love this new technology."}},
            headers=HEADERS,
        )

    def test_returns_all_scores_and_dominant_emotion(self, mock_post):
        mock_post.return_value = make_response(
            anger=0.01, disgust=0.02, fear=0.03, joy=0.9, sadness=0.04)

        result = emotion_detector("I love this new technology.")

        self.assertEqual(set(result), EXPECTED_KEYS)
        self.assertEqual(result['anger'], 0.01)
        self.assertEqual(result['disgust'], 0.02)
        self.assertEqual(result['fear'], 0.03)
        self.assertEqual(result['joy'], 0.9)
        self.assertEqual(result['sadness'], 0.04)
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_dominant_emotion_for_each_emotion(self, mock_post):
        cases = [
            ("I am glad this happened", 'joy'),
            ("I am really mad about this", 'anger'),
            ("I feel disgusted just hearing about this", 'disgust'),
            ("I am so sad about this", 'sadness'),
            ("I am really afraid that this will happen", 'fear'),
        ]
        for text, emotion in cases:
            with self.subTest(text=text):
                scores = {e: 0.1 for e in ('anger', 'disgust', 'fear', 'joy', 'sadness')}
                scores[emotion] = 0.8
                mock_post.return_value = make_response(**scores)

                result = emotion_detector(text)

                self.assertEqual(result['dominant_emotion'], emotion)

    def test_tie_picks_first_emotion_in_response_order(self, mock_post):
        mock_post.return_value = make_response(
            anger=0.5, disgust=0.1, fear=0.5, joy=0.1, sadness=0.1)

        result = emotion_detector("Mixed feelings")

        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_malformed_response_raises(self, mock_post):
        response = MagicMock()
        response.text = json.dumps({"unexpected": "payload"})
        mock_post.return_value = response

        with self.assertRaises(KeyError):
            emotion_detector("Anything")


@unittest.skipUnless(os.environ.get('RUN_LIVE_TESTS') == '1',
                     'Set RUN_LIVE_TESTS=1 to call the real Watson service')
class TestEmotionDetectorLive(unittest.TestCase):
    """Integration tests against the real Watson NLP service."""

    def test_dominant_emotions(self):
        cases = [
            ("I am glad this happened", 'joy'),
            ("I am really mad about this", 'anger'),
            ("I feel disgusted just hearing about this", 'disgust'),
            ("I am so sad about this", 'sadness'),
            ("I am really afraid that this will happen", 'fear'),
        ]
        for text, emotion in cases:
            with self.subTest(text=text):
                self.assertEqual(emotion_detector(text)['dominant_emotion'], emotion)


if __name__ == '__main__':
    unittest.main()
