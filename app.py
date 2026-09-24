"""Run with: streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from hotel_service import HotelService, friendly_error, hotel_catalog
from email_creative import render_email, export_email, validate_booking_url

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=True)
st.set_page_config(page_title='Guest Notes · Hotel Review Studio', page_icon='✦', layout='wide')
st.markdown('''<style>
.stApp {background: #0e1418; color: #eef4f1;}
h1, h2, h3 {letter-spacing: -0.035em;}
[data-testid="stSidebar"] {background: #182229;}
.stButton>button[kind="primary"] {background: #244c43; border-color: #244c43;}
[data-testid="stChatMessage"] {background: #182229; border-radius: 14px;}
</style>''', unsafe_allow_html=True)


@st.cache_data
def catalog(path, modified):
    return hotel_catalog(path)


def source_list(sources):
    with st.expander(f'Review evidence · {len(sources)} excerpts'):
        for i, source in enumerate(sources, start=1):
            st.write(f"[{i}] {source.get('hotel_name', '')} · {source.get('review_date', '')}")
            st.caption(source.get('review_title', ''))
            st.write(source.get('text', ''))
            url = source.get('source_url', '')
            if url.startswith(('https://', 'http://')):
                st.link_button('Review source', url)


st.caption('GUEST NOTES / HOTEL REVIEW STUDIO')
st.title('Turn guest experiences into your next great story.')
st.write('Explore what guests say. Bring the best of your hotel to life.')

with st.sidebar:
    st.header('Review library')
    data_path = Path(os.getenv('HOTEL_REVIEWS_PATH', str(ROOT / 'hotel_reviews.json'))).expanduser()
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists() and 'HOTEL_REVIEWS_PATH' not in os.environ:
        data_path = ROOT / 'Hotel Reviews.json'
    try:
        hotels = catalog(str(data_path), data_path.stat().st_mtime_ns)
        if not hotels:
            raise ValueError('No hotel names and IDs were found in the review file.')
    except (OSError, ValueError) as exc:
        st.error('Cannot load the hotel list. Place Hotel Reviews.json beside app.py or set HOTEL_REVIEWS_PATH.')
        st.stop()
    if st.button('Connect / refresh reviews', use_container_width=True):
        for key in ('service', 'collections', 'active_scope', 'messages', 'analysis', 'image', 'email_copy', 'promo_hotel'):
            st.session_state.pop(key, None)
        try:
            keys = [os.getenv(name, '').strip() for name in
                    ('OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME')]
            if any(not value or value.startswith(('your-', 'sk-your-')) for value in keys):
                raise ValueError('Set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX_NAME in .env.')
            with st.spinner('Connecting to your reviews…'):
                service = HotelService(*keys)
                collections = service.namespaces()
            if not collections:
                raise ValueError('No indexed review collections found. Complete notebook indexing, then reconnect.')
            st.session_state.service = service
            st.session_state.collections = collections
        except Exception as exc:
            st.error(friendly_error(exc))
    collections = st.session_state.get('collections', {})
    namespace = st.selectbox('Review collection', sorted(collections),
                             help='Choose the namespace printed by the notebook’s chunking cell.') if collections else None
    if namespace:
        st.caption(f'{collections[namespace]:,} review chunks in this collection')
    st.caption('Finish indexing in hotel_reviews_rag.ipynb before connecting. API requests need available OpenAI credits.')

scope = namespace
if 'messages' not in st.session_state or st.session_state.get('active_scope') != scope:
    st.session_state.active_scope = scope
    st.session_state.messages = []
    st.session_state.pop('analysis', None)
    st.session_state.pop('image', None)
    st.session_state.pop('email_copy', None)

service = st.session_state.get('service')
ready = service is not None and namespace is not None
chat_tab, promo_tab = st.tabs(['01  ·  Ask the reviews', '02  ·  Create a promotion'])

with chat_tab:
    st.subheader('What would you like to discover?')
    st.caption('Ask about any hotel, compare guest experiences, or explore what travelers value.')
    if not ready:
        st.info('Connect to your indexed reviews in the sidebar to start exploring.')
    if st.button('Clear conversation'):
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
            if message.get('sources'):
                source_list(message['sources'])
    question = st.chat_input('Ask anything about the hotel reviews…', disabled=not ready, max_chars=2000)
    if question:
        history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
        with st.chat_message('user'):
            st.write(question)
        try:
            with st.chat_message('assistant'):
                with st.spinner('Reading relevant guest reviews…'):
                    result = service.answer(question, namespace, None, history)
                st.write(result['answer'])
                if result['sources']:
                    source_list(result['sources'])
            st.session_state.messages.extend([
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': result['answer'], 'sources': result['sources']},
            ])
        except Exception as exc:
            st.error(friendly_error(exc))
            st.caption('Your question was not saved. Submit it again once the issue is resolved.')

with promo_tab:
    st.subheader('An escape inspired by your guests.')
    st.write('Create an email campaign for a **2 days, 3 nights** hotel promotion — with copy and imagery inspired by what customers love.')
    selected = st.selectbox(
        'Select a hotel', hotels, key='promotion_hotel_selector',
        format_func=lambda h: f"{h['name']} · {h['city']} · {h['id'][-6:]}",
    )
    if selected != st.session_state.get('promo_hotel'):
        st.session_state.promo_hotel = selected
        for key in ('analysis', 'image', 'email_copy'):
            st.session_state.pop(key, None)
    if selected:
        st.success(f"Creating for {selected['name']} · {selected['city']}")
    st.caption('The creative uses three positive themes from retrieved reviews, not a statistical ranking of every review.')
    if st.button('Create email copy from reviews', type='primary', disabled=not ready or not selected):
        for key in ('analysis', 'image', 'email_copy'):
            st.session_state.pop(key, None)
        try:
            with st.spinner('Finding guest favorites and writing your email…'):
                analysis = service.positive_features(namespace, selected)
                copy = service.create_email(selected, analysis['features'])
                st.session_state.analysis = analysis
                st.session_state.email_copy = copy
        except Exception as exc:
            st.error(friendly_error(exc))
    analysis = st.session_state.get('analysis')
    copy = st.session_state.get('email_copy')
    if analysis and copy:
        with st.expander('The guest insights behind this campaign'):
            for feature in analysis['features']:
                st.subheader(feature['title'])
                st.write(feature['description'])
                st.caption('Review sources: ' + ', '.join(f'[{i}]' for i in feature['sources']))
            source_list(analysis['sources'])
        st.write('**Subject:** ' + copy['subject'])
        st.caption('Preview text: ' + copy['preheader'])
        booking_url = st.text_input('Booking or enquiry URL (optional)', placeholder='https://…', max_chars=2000)
        valid_url = True
        try:
            validate_booking_url(booking_url)
        except ValueError as exc:
            st.error(str(exc))
            valid_url = False
        style = st.radio('Image direction', ['Warm editorial illustration', 'Minimal luxury', 'Vibrant travel illustration'], horizontal=True)
        st.caption('The hero image will reflect the same three guest favorites as the email copy.')
        if st.button('Generate matching email image', type='primary'):
            try:
                with st.spinner('Creating your email hero image…'):
                    content = service.create_image(selected, analysis['features'], style, '1536x1024', email_copy=copy)
                st.session_state.image = {'bytes': content, 'style': style}
            except Exception as exc:
                st.error(friendly_error(exc))
        image = st.session_state.get('image', {}).get('bytes')
        if valid_url:
            html = render_email(selected, analysis['features'], copy, image, booking_url)
            components.html(html, height=1000, scrolling=True)
            st.download_button('Download email HTML', html, 'hotel-promotion.html', 'text/html')
            st.download_button('Download email draft (.eml)', export_email(selected, analysis['features'], copy, image, booking_url), 'hotel-promotion.eml', 'message/rfc822')
            if image:
                st.download_button('Download hero image', image, 'hotel-promotion.png', 'image/png')
            st.caption('The email draft embeds the image for email clients. HTML includes an inline image for browser previews. No email is sent.')
    else:
        st.info('Select a hotel, then create the email from its guest reviews.')
