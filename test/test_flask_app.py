import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>VeriNews - Fake News Detection</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data={
            'title': 'Breaking: Scientists discover new planet',
            'source_domain': 'reuters.com',
            'tweet_num': '150'
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'FAKE NEWS' in response.data or b'REAL NEWS' in response.data,
            "Response should contain either 'FAKE NEWS' or 'REAL NEWS'"
        )

if __name__ == '__main__':
    unittest.main()