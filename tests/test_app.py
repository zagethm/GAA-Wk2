import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from hotel_service import Feature, HotelService, Promotion, friendly_error, hotel_catalog

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aAnkAAAAASUVORK5CYII=')
HOTEL = {'id': 'hotel-1', 'name': 'Test Hotel', 'city': 'Boston'}
DOC = {'hotel_name': 'Test Hotel', 'review_id': 'review-1', 'vector_id': 'vector-1',
       'text': 'Clean rooms, friendly staff and a convenient location.', 'review_date': '2018', 'score': 0.9}
FEATURES = [Feature(title=t, description='Guests praised ' + t, sources=[1])
            for t in ['Clean rooms', 'Friendly staff', 'Convenient location']]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = HotelService.__new__(HotelService)
        self.service.ai = Mock()
        self.service.index = Mock()

    def test_retrieval_uses_notebook_dimensions_namespace_and_hotel(self):
        self.service.ai.embeddings.create.return_value = NS(data=[NS(embedding=[0.0] * 512)])
        self.service.index.query.return_value = NS(matches=[NS(id='vector-1', score=.9, metadata=DOC)])
        result = self.service.retrieve('Clean rooms?', 'hotel-reviews-test', HOTEL)
        kwargs = self.service.index.query.call_args.kwargs
        self.assertEqual(len(kwargs['vector']), 512)
        self.assertEqual(kwargs['namespace'], 'hotel-reviews-test')
        self.assertEqual(kwargs['filter']['hotel_id'], {'$eq': 'hotel-1'})
        self.assertEqual(result[0]['text'], DOC['text'])

    def test_no_results_does_not_generate_answer(self):
        self.service.retrieve = Mock(return_value=[])
        result = self.service.answer('Question', 'namespace', HOTEL, [])
        self.assertEqual(result['sources'], [])
        self.service.ai.chat.completions.create.assert_not_called()

    def test_open_ended_search_has_no_hotel_filter(self):
        self.service.ai.embeddings.create.return_value = NS(data=[NS(embedding=[0.0] * 512)])
        self.service.index.query.return_value = NS(matches=[])
        self.service.retrieve('Which hotels have friendly staff?', 'namespace')
        self.assertNotIn('filter', self.service.index.query.call_args.kwargs)

    def test_email_export_preserves_offer_escapes_html_and_embeds_image(self):
        from email_creative import render_email, export_email, validate_booking_url
        from email import message_from_bytes
        copy = {'subject': 'A stay\nfor you', 'preheader': 'Preview', 'headline': '<script>bad</script>',
                'introduction': 'A lovely escape', 'closing': 'See you soon', 'cta': 'Enquire'}
        features = [f.model_dump() for f in FEATURES]
        html = render_email(HOTEL, features, copy, PNG)
        self.assertIn('2 days, 3 nights', html)
        self.assertNotIn('<script>', html)
        message = message_from_bytes(export_email(HOTEL, features, copy, PNG))
        self.assertEqual(message['Subject'], 'A stay for you')
        self.assertTrue(any(p.get('Content-ID') == '<hotel-hero>' for p in message.walk()))
        with self.assertRaises(ValueError):
            validate_booking_url('javascript:alert(1)')

    def test_follow_up_rewrite_and_grounded_answer(self):
        self.service.retrieve = Mock(return_value=[DOC])
        self.service.ai.chat.completions.create.side_effect = [
            NS(choices=[NS(message=NS(content='Are the hotel rooms clean?'))]),
            NS(choices=[NS(message=NS(content='Guests liked clean rooms [1].'))]),
        ]
        result = self.service.answer('And the rooms?', 'namespace', HOTEL,
                                     [{'role': 'user', 'content': 'How is Test Hotel?'}])
        self.assertIn('[1]', result['answer'])
        self.service.retrieve.assert_called_with('Are the hotel rooms clean?', 'namespace', HOTEL)

    def test_features_deduplicate_reviews_and_validate_evidence(self):
        self.service.retrieve = Mock(return_value=[DOC, {**DOC, 'vector_id': 'another-chunk'}])
        response = NS(choices=[NS(message=NS(parsed=Promotion(features=FEATURES)))])
        self.service.ai.beta.chat.completions.parse.return_value = response
        result = self.service.positive_features('namespace', HOTEL)
        self.assertEqual(len(result['sources']), 1)
        self.assertEqual(len(result['features']), 3)
        response.choices[0].message.parsed = Promotion(features=FEATURES[:2])
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            self.service.positive_features('namespace', HOTEL)
        bad = Feature(title='Invalid', description='Unsupported source', sources=[99])
        response.choices[0].message.parsed = Promotion(features=[*FEATURES[:2], bad])
        with self.assertRaisesRegex(ValueError, 'invalid source'):
            self.service.positive_features('namespace', HOTEL)

    def test_image_decoding_and_prompt(self):
        self.service.ai.with_options.return_value.images.generate.return_value = NS(
            data=[NS(b64_json=base64.b64encode(PNG).decode())])
        image = self.service.create_image(HOTEL, [f.model_dump() for f in FEATURES], 'Minimal', '1024x1024')
        self.assertEqual(image, PNG)
        request = self.service.ai.with_options.return_value.images.generate.call_args.kwargs
        self.assertEqual(request['model'], 'gpt-image-1')
        for feature in FEATURES:
            self.assertIn(feature.title, request['prompt'])

    def test_quota_error_and_catalog(self):
        error = RuntimeError('You have no credits remaining. SECRET')
        error.status_code = 429
        self.assertIn('Add credits', friendly_error(error))
        self.assertNotIn('SECRET', friendly_error(error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reviews.json'
            path.write_text(json.dumps([{'id': '1', 'name': 'Hotel', 'city': 'Boston'}] * 2))
            self.assertEqual(len(hotel_catalog(path)), 1)


class UIWorkflowTests(unittest.TestCase):
    def test_both_tabs_chat_promotion_download_and_hotel_reset(self):
        from streamlit.testing.v1 import AppTest
        service = Mock()
        service.namespaces.return_value = {'hotel-reviews-test': 100}
        service.answer.return_value = {'answer': 'Guests liked clean rooms [1].', 'sources': [DOC]}
        service.positive_features.return_value = {'features': [f.model_dump() for f in FEATURES], 'sources': [DOC]}
        service.create_image.return_value = PNG
        service.create_email.return_value = {'subject': 'Your escape', 'preheader': 'Guest favorites', 'headline': 'Stay a little longer', 'introduction': 'Enjoy 2 days, 3 nights.', 'closing': 'Plan your escape.', 'cta': 'Enquire now'}
        hotels = [HOTEL, {**HOTEL, 'id': 'hotel-2', 'name': 'Another Hotel'}]
        with patch('hotel_service.HotelService', return_value=service), \
             patch('hotel_service.hotel_catalog', return_value=hotels), \
             patch('dotenv.load_dotenv'), \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test', 'PINECONE_API_KEY': 'test', 'PINECONE_INDEX_NAME': 'test'}):
            app = AppTest.from_file('app.py').run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 2)
            self.assertTrue(app.chat_input[0].disabled)
            app.sidebar.button[0].click().run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value('Are rooms clean?').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['messages']), 2)
            app.selectbox(key='promotion_hotel_selector').select(hotels[0]).run()
            next(b for b in app.button if b.label == 'Create email copy from reviews').click().run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == 'Generate matching email image').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['image']['bytes'], PNG)
            app.run()
            service.create_image.assert_called_once()
            self.assertEqual(len(app.get('download_button')), 3)
            self.assertTrue(all(s.label != 'Select a hotel' for s in app.sidebar.selectbox))
            app.selectbox(key='promotion_hotel_selector').select(hotels[1]).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['messages']), 2)
            self.assertNotIn('image', app.session_state)


if __name__ == '__main__':
    unittest.main()
