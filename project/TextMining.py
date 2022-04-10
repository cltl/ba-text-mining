import requests
import re
import numpy
import spacy
nlp = spacy.load('en_core_web_sm')

def load_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

try:
    vader_model = load_vader()
except LookupError as e:
    print(e)
    print('Attempting to download automatically...')
    import nltk 
    nltk.download('vader_lexicon')
    vader_model = load_vader()
    print('Success')


api_url = 'https://api.twitter.com/2/tweets/search/recent'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIyUagEAAAAAwJxB%2BsJcNlgKNgKMKifUDwQXng4%3DXRKHHBl5MeXoeYlDra1KYstSRKM3D3WjTsaBt2cgktFOJYplTe'

def search_twitter(query, bearer_token, results: int = 10):
    payload = {
        'query': 'lang:en '+str(query),
        'max_results': results,
        'sort_order': 'relevancy',
        'tweet.fields': 'id,text,public_metrics'
    }

    auth = {
        'Authorization': 'Bearer '+str(bearer_token)
    }

    response = requests.get(api_url, params=payload, headers=auth)
    if response.status_code != 200:
        print('Error: Status Code '+str(response.status_code))
        return
    if not 'data' in response.json():
        print('Error: No tweets found')
        return
    return response.json()

def get_replies(id, bearer_token, results: int = 10):
    payload = {
        'query': 'lang:en conversation_id:'+str(id),
        'max_results': results,
        'sort_order': 'relevancy',
        'tweet.fields': 'text,public_metrics'
    }

    auth = {
        'Authorization': 'Bearer '+str(bearer_token)
    }

    response = requests.get(api_url, params=payload, headers=auth)
    if response.status_code != 200:
        print('Error: Status Code '+str(response.status_code)+': '+response.text)
        return
    if not 'data' in response.json():
        #print('Error: No replies found')
        return
    return response.json()

def analyze_topic(
    query, 
    bearer_token, 
    minimum_likes = 5,
    num_parents: int = 10, 
    num_replies: int = 10,
    parent_weight = 1,
    reply_weight = 1
):
    tweets = search_twitter(query, bearer_token, num_parents)
    return analyze_tweets(
        tweets,
        minimum_likes=minimum_likes,
        num_replies=num_replies,
        parent_weight=parent_weight,
        reply_weight=reply_weight
    )

def analyze_tweets(
    tweets, 
    minimum_likes = 5,
    num_replies: int = 10,
    parent_weight = 1,
    reply_weight = 1
):
    if tweets and 'data' in tweets:
        sentiments = []
        for tweet in tweets['data']:
            if tweet['public_metrics']['like_count'] >= minimum_likes:
                s = sentiment(tweet['text'])
                if s:
                    sentiments.append(
                            {
                                'sentiment': s,
                                'weight': parent_weight*(1 + tweet['public_metrics']['retweet_count'] + tweet['public_metrics']['quote_count'] + tweet['public_metrics']['like_count'])
                            }
                        )

            if num_replies:
                replies = get_replies(tweet['id'], bearer_token, num_replies)
                if replies and 'data' in replies:
                    for reply in replies['data']:
                        metrics = reply['public_metrics']
                        if metrics['like_count'] >= minimum_likes:
                            s = sentiment(reply['text'])
                            if s:
                                sentiments.append(
                                    {
                                        'sentiment': s,
                                        'weight': reply_weight*(1 + metrics['retweet_count'] + metrics['quote_count'] + metrics['like_count'])
                                    }
                                )

        weighted_average = numpy.average(
            [item['sentiment'] for item in sentiments],
            weights = [item['weight'] for item in sentiments]
        )
        return weighted_average

def sentiment_to_sentence(value, min=-1, max=1):
    if value < min+(max-min)*0.17:
        return "strongly negative"
    elif value < min+(max-min)*0.33:
        return "moderately negative"
    elif value < min+(max-min)*0.5:
        return "slightly negative"
    elif value < min+(max-min)*0.67:
        return "slightly positive"
    elif value < min+(max-min)*0.83:
        return "moderately positive"
    else:
        return "strongly positive"

def sentiment_to_line(value, min=-1, max=1):
    line = '----------|----------'
    pos = (value-min) / (max-min)
    pos = int(numpy.round(pos*len(line)))
    line_val = line[:pos]+'*'+(line[(pos+1):] if pos<len(line) else '')
    return '(-) ['+line_val+'] (+)'

"""
#code below is to implement VADER with NLTK. Doesn't work yet though.    
def run_spacey(textual_unit, 
              lemmatize=False, 
              parts_of_speech_to_consider=None,
              verbose=False):

    doc = nlp(textual_unit)
        
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-': 
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add) 
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))
    
    if verbose:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores['compound']
"""
    
def sentiment(text):
    scores = vader_model.polarity_scores(text)
    return scores['compound']

print("""Welcome to
    ____             __  __       _    __          __         
   / __ \____ ______/ /_/ /_     | |  / /___ _____/ /__  _____
  / / / / __ `/ ___/ __/ __ \    | | / / __ `/ __  / _ \/ ___/
 / /_/ / /_/ / /  / /_/ / / /    | |/ / /_/ / /_/ /  __/ /    
/_____/\__,_/_/   \__/_/ /_/     |___/\__,_/\__,_/\___/_/     """)
while True:
    print("""
What would you like to do?
[1] Analyize a topic
[2] Analyze tweet replies
[3] Analyze arbitrary text
[0] Exit""")
    mode = str(input('=> '))

    if mode=='1':
        topic = '"'+input('Enter a topic to analyze: ').strip()+'"'
        num_p = input('How many parent tweets to get? (10-100, default=100) ')
        num_p = num_p if num_p else 100
        num_r = input('How many reply tweets to get? (10-100, default=0) ')
        num_r = num_r if num_r else 0

        print('Analyzing topic...')
        tweets = search_twitter(topic, bearer_token, num_p)
        snt = analyze_tweets(tweets, num_replies=num_r)
    elif mode=='2':
        match = None
        while not match:
            parent = input('Enter URL of parent tweet: ')
            match = re.match(r'(?:https?:\/\/)?(?:www.)?twitter.com/[^/]*/status/([0-9]*)', parent)
            if not match:
                print('Error: Not a valid tweet URL')
        conversation_id = match.group(1)
        num_r = input('How many reply tweets to get? (10-100, default=25) ')
        num_r = num_r if num_r else 25

        print('Analyzing replies...')
        tweets = get_replies(conversation_id, bearer_token, num_r)
        snt = analyze_tweets(tweets, num_replies=0)
    elif mode=='3':
        filename = input('Enter path to file: ')
        with open(filename) as file:
            text = file.read()
        doc = nlp(text)
        snt = numpy.average([sentiment(str(s)) for s in doc.sents])
    else:
        break

    if snt:
        print('Done! Found overall ' + sentiment_to_sentence(snt) + ' sentiment (' + str(numpy.round(snt, 4))+')')
        print(sentiment_to_line(snt))