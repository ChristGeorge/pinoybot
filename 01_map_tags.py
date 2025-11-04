def extract_special_token_features(word): # I renamed the function
    features = {}

    #numbers
    features['is_number'] = bool(re.match(r'^[0-9]+([.,][0-9]+)*$', word))
    features['has_digit'] = bool(re.search(r"[0-9]", word))

    #capitalization
    features['is_first_cap'] = bool(re.match(r"^[A-Z]", word))
    features['is_all_caps'] = bool(re.match(r"^[A-Z]{2,}$", word))
    features['is_all_lower'] = word.islower()

    #symbols
    features['is_punctuation'] = bool(re.match(r'^[^a-zA-Z0-9\s]+$', word))
    features['has_special_char'] = bool(re.search(r'[^a-zA-Z0-9\s]', word)) 

    #alphanumeric patterns
    features['is_alphanumeric'] = bool(re.search(r"[a-zA-Z]", word) and re.search(r"[0-9]", word))

    #length
    features['word_length'] = len(word)
    features['is_very_short'] = bool(len(word) <= 2)

    return features


def extract_all_features(word):
    #combines all features into one dictionary
    features = {}

    features.update(extract_filipino_affix_features(word))
    features.update(extract_english_affixes(word))
    features.update(extract_character_features(word))
    features.update(extract_special_token_features(word))

    return features
    

#use the function in apply to determine the tag and creates a panda series (kind of like a list)
tag = df.apply(lambda row: label_is_true(row), axis=1)

#creates a new column that will contain either "ENG", "FIL", or "OTH"
df['three_class_label'] = tag.apply(map_to_three_classes)

# --- Test for Task 2.5 ---
#if __name__ == "__main__":
    #print("\nTesting Master Feature Extractor:")
    #test_words = ['naglunch', 'kumain', 'corrupt', 'playing', 'Manila', '.', '2023']

    #for word in test_words:
        #features = extract_all_features(word)
        #print(f"\nWord: '{word}' ({len(features)} features extracted)")

        # Let's just show a few features to prove it works
        #print(f"  ... has_nag_prefix: {features.get('has_nag_prefix')}")
        #print(f"  ... has_ing_suffix: {features.get('has_ing_suffix')}")
        #print(f"  ... vowel_ratio: {features.get('vowel_ratio')}")
        #print(f"  ... is_number: {features.get('is_number')}")

# Creates a new CVS file named "processed_data.csv" to be read by the training model.
print("PHASE 2: FILE 'map_tags.py':")
df.to_csv("processed_data.csv", index=False)
print("Data saved as 'processed_data.csv'.")
print(f"Total rows: {len(df)}")
