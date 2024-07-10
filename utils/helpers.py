import nltk
from nltk import ngrams, pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")
google_search_key = os.getenv("GOOGLE_PROGRAMMABLE_SEARCH_KEY")

client = OpenAI(api_key=open_ai_key)

nltk.download("punkt")


def generate_ngram_results(set_id, df, count_threshold=3):

    # Filter DataFrame for current EssaySet and score1 == 2
    filtered_df = df[(df["EssaySet"] == set_id) & (df["Score1"] == 2)]

    # Print progress for each EssaySet
    print(f"Processing EssaySet {set_id} with {len(filtered_df)} essays.")

    # Initialize a Counter to count n-grams across all essays in the set
    ngram_counter = Counter()

    # Process each CorrectedSpellingEssayText in the filtered_df DataFrame
    for essay in filtered_df["CorrectedSpellingEssayText"]:
        tokens = word_tokenize(essay)
        pos_tags = pos_tag(tokens)

        # Generate bi-grams, tri-grams, and tetra-grams
        bi_grams = list(ngrams(pos_tags, 2))
        tri_grams = list(ngrams(pos_tags, 3))
        tetra_grams = list(ngrams(pos_tags, 4))

        # Concatenate all n-grams into a single list and update the counter
        all_ngrams = bi_grams + tri_grams + tetra_grams
        ngram_counter.update(all_ngrams)

    # Filter n-grams that appeared at least 3 times
    frequent_ngrams = [
        ngram for ngram, count in ngram_counter.items() if count >= count_threshold
    ]

    # Print after processing each set
    print(
        f"Completed processing EssaySet {set_id}, with {len(frequent_ngrams)} n-grams."
    )

    return frequent_ngrams


def extract_articles(keyword):
    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "key": google_search_key,
        "cx": "901399343dffc41a5",
        "q": keyword,
        "start": 1,
        "num": 10,
        "lr": "lang_en",
        "filter": 1,
    }
    articles = []
    all_links = []
    try:
        for i in range(2):
            response = requests.get(url, params={**params, "start": i * 10 + 1})
            data = response.json()
            links = [item["link"] for item in data["items"] if "link" in item]
            all_links.extend(links)

        print("All links: ", all_links)

        # Function to scrape a single link
        def scrape_link(link):
            try:
                page = requests.get(link, timeout=10)
                soup = BeautifulSoup(page.content, "html.parser")
                return soup.get_text()
            except Exception as e:
                print(f"Failed to scrape {link}: {e}")
                return None

        # Use ThreadPoolExecutor to scrape links concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(scrape_link, all_links))

        # Filter out None results and extend articles list
        articles.extend([result for result in results if result is not None])

    except Exception as e:
        print(f"An error occurred: {e}")
        # all.extend(data["items"])

    return articles


def extract_domain_specific_keywords(prompt, answers):
    # first 50k tokens
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(answers)
    first_50000_tokens = tokens[:50000]
    extracted_text = tokenizer.decode(first_50000_tokens)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {
                "role": "user",
                "content": f"Here is a question: {prompt}. Identify 100 domain-specific keywords that are most relevant to the question from the following set of student answers, and return them in a JSON object with an array field named 'keywords':\n\n{extracted_text}",
            },
        ],
    )

    keywords = json.loads(response.choices[0].message.content)

    return keywords


def extract_tfidf_from_articles(keyword, articles):
    def preprocess(text):
        return text.lower()

    # Preprocess articles
    preprocessed_articles = [preprocess(article) for article in articles]

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the articles
    tfidf_matrix = vectorizer.fit_transform(preprocessed_articles)

    # Get the feature names (i.e., the words)
    feature_names = vectorizer.get_feature_names_out()

    # Split the keyword into individual words and lower each
    keywords = keyword.lower().split()

    # Initialize list to store tf-idf values for each word in the keyword
    tfidf_values_list = []

    # Iterate over each word in the keyword
    for word in keywords:
        try:
            # Find the index of the word
            word_index = feature_names.tolist().index(word)
            # Get the tf-idf values for the word across all documents
            tfidf_values = tfidf_matrix[:, word_index].toarray().flatten()
            # Append the average tf-idf value for the word
            tfidf_values_list.append(tfidf_values.mean())
        except ValueError:
            # If the word is not found, append 0.0
            tfidf_values_list.append(0.0)

    # Calculate the average tf-idf value for the entire keyword phrase
    if tfidf_values_list:
        average_tfidf = sum(tfidf_values_list) / len(tfidf_values_list)
    else:
        average_tfidf = 0.0

    return average_tfidf


def extract_weighted_keywords(setno, df):
    all_articles = {}

    # Load the prompt inside the function
    prompt = load_prompt(setno)

    answers = " ".join(
        df[df["EssaySet"] == setno]["CorrectedSpellingEssayText"].tolist()
    )
    keywords = extract_domain_specific_keywords(prompt, answers)["keywords"]

    set_scores = {}

    total_keywords = len(keywords)
    for i, keyword in enumerate(keywords, 1):
        articles = extract_articles(keyword)
        tfidf = extract_tfidf_from_articles(keyword, articles)

        set_scores[keyword] = tfidf
        print(f"Progress: {i}/{total_keywords} - keyword: {keyword}, tfidf: {tfidf}")

        if setno not in all_articles:
            all_articles[setno] = {}
        all_articles[setno][keyword] = articles

    return set_scores, all_articles


def load_prompt(set_id):
    file_name = f"prompts/asap_{set_id:02d}.txt"
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            prompt = file.read().strip()
    else:
        print(f"Prompt file {file_name} not found.")
        prompt = None
    return prompt


def load_only_prompt(set_id):
    file_name = f"only_prompts/asap_{set_id:02d}.txt"
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            prompt = file.read().strip()
    else:
        print(f"Prompt file {file_name} not found.")
        prompt = None
    return prompt


from numpy import hstack


# run this for everythting ... so i can just do it once, and then save it as a csv to re-load ...
def add_features_to_df(
    df,
    feature_extractor,
    use_better_spelling=True,
    add_word2vec=False,
    add_doc2vec=False,
    add_pos=False,
    add_prompt_overlap=False,
    add_weighted_keywords=False,
    add_lexical_overlap=False,
    add_stylistic_features=False,
    add_logical_operators=False,
    add_temporal_features=False,
):
    if use_better_spelling:
        key = "CorrectedSpellingEssayText"
    else:
        key = "EssayText"

    if add_word2vec:
        df["word2vec_features"] = None
    if add_doc2vec:
        df["doc2vec_features"] = None
    if add_pos:
        df["pos_features"] = None
    if add_prompt_overlap:
        df["prompt_overlap_features"] = None
    if add_weighted_keywords:
        df["weighted_keywords_features"] = None
    if add_lexical_overlap:
        df["lexical_overlap_features"] = None
    if add_stylistic_features:
        df["stylistic_features"] = None
    if add_logical_operators:
        df["logical_operators_features"] = None
    if add_temporal_features:
        df["temporal_features"] = None

    total_items = len(df)
    print(f"Processing {total_items} items...")

    # Iterate over each row in the training DataFrame
    num = 0
    for index, row in df.iterrows():
        # Extract features using the feature_extractor object
        if add_word2vec:
            word2vec_features = feature_extractor.word2vec(row[key])
            df.at[index, "word2vec_features"] = word2vec_features
        if add_doc2vec:
            doc2vec_features = feature_extractor.doc2vec(row[key])
            df.at[index, "doc2vec_features"] = doc2vec_features
        if add_pos:
            pos_features = feature_extractor.pos(row[key])
            df.at[index, "pos_features"] = pos_features
        if add_prompt_overlap:
            prompt_overlap_features = feature_extractor.prompt_overlap(row[key])
            df.at[index, "prompt_overlap_features"] = prompt_overlap_features
        if add_weighted_keywords:
            weighted_keywords_features = (
                feature_extractor.weighted_domain_specific_keywords(row[key])
            )
            df.at[index, "weighted_keywords_features"] = weighted_keywords_features
        if add_lexical_overlap:
            lexical_overlap_features = feature_extractor.lexical_overlap(row[key])
            df.at[index, "lexical_overlap_features"] = lexical_overlap_features
        if add_stylistic_features:
            stylistic_features = feature_extractor.stylistic_features(row[key])
            df.at[index, "stylistic_features"] = stylistic_features
        if add_logical_operators:
            logical_operators_features = feature_extractor.logical_operators(row[key])
            df.at[index, "logical_operators_features"] = logical_operators_features
        if add_temporal_features:
            temporal_features = feature_extractor.temporal_features(row[key])
            df.at[index, "temporal_features"] = temporal_features

        # Update after every 1000 items
        if (num) % 1000 == 0:
            print(f"Processed {num} items.")
        num += 1

    return df
