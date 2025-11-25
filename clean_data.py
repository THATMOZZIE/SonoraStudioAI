# ========================================
# INSTAGRAM DATA CLEANING PIPELINE
# ========================================
import pandas as pd
import numpy as np
import ast
from datetime import datetime

print("📊 Instagram Data Cleaning Pipeline")
print("="*70)

# Load raw data
print("\n🔄 Loading raw data...")
df_raw = pd.read_csv('instagram_raw.csv', low_memory=False)
print(f"✅ Loaded {len(df_raw)} rows with {len(df_raw.columns)} columns")

# Extract core fields
print("\n🔄 Extracting core fields...")
df_clean = pd.DataFrame()

direct_columns = {
    'id': 'id', 'type': 'type', 'shortCode': 'shortCode', 'caption': 'caption',
    'url': 'url', 'likesCount': 'likesCount', 'commentsCount': 'commentsCount',
    'videoViewCount': 'videoViewCount', 'videoPlayCount': 'videoPlayCount',
    'ownerUsername': 'ownerUsername', 'ownerFullName': 'ownerFullName',
    'ownerId': 'ownerId', 'displayUrl': 'displayUrl', 'videoUrl': 'videoUrl',
    'videoDuration': 'videoDuration', 'dimensionsHeight': 'dimensionsHeight',
    'dimensionsWidth': 'dimensionsWidth', 'locationName': 'locationName',
    'locationId': 'locationId', 'isSponsored': 'isSponsored',
    'isPinned': 'isPinned', 'isCommentsDisabled': 'isCommentsDisabled',
    'productType': 'productType', 'alt': 'alt', 'inputUrl': 'inputUrl',
    'firstComment': 'firstComment', 'timestamp': 'timestamp'
}

for new_col, old_col in direct_columns.items():
    df_clean[new_col] = df_raw.get(old_col)

# Numeric conversions
for col in ['likesCount', 'commentsCount', 'videoViewCount', 'videoPlayCount',
            'dimensionsHeight', 'dimensionsWidth']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

df_clean['videoDuration'] = pd.to_numeric(df_clean.get('videoDuration', 0), errors='coerce').fillna(0)

# Boolean conversions
for col in ['isSponsored', 'isPinned', 'isCommentsDisabled']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(False).astype(bool)

# Fill text columns
for col in ['caption', 'ownerUsername', 'ownerFullName', 'productType', 'locationName']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('').astype(str)

print(f"✅ Core fields extracted")

# Parse nested arrays
print("\n🔄 Parsing hashtags, mentions, and tagged users...")

hashtag_cols = [col for col in df_raw.columns if col.startswith('hashtags/')]
mention_cols = [col for col in df_raw.columns if col.startswith('mentions/')]
tagged_cols = [col for col in df_raw.columns if 'taggedUsers/' in col and col.endswith('/username')]
coauthor_cols = [col for col in df_raw.columns if 'coauthorProducers/' in col and col.endswith('/username')]
image_cols = [col for col in df_raw.columns if col.startswith('images/')]

def extract_unique_values(row, columns):
    values = []
    for col in columns:
        val = row.get(col)
        if pd.notna(val):
            cleaned = str(val).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values

df_clean['hashtags'] = df_raw[hashtag_cols].apply(lambda row: str(extract_unique_values(row, hashtag_cols)), axis=1) if hashtag_cols else '[]'
df_clean['mentions'] = df_raw[mention_cols].apply(lambda row: str(extract_unique_values(row, mention_cols)), axis=1) if mention_cols else '[]'
df_clean['taggedUsers'] = df_raw[tagged_cols].apply(lambda row: str(extract_unique_values(row, tagged_cols)), axis=1) if tagged_cols else '[]'
df_clean['coauthorProducers'] = df_raw[coauthor_cols].apply(lambda row: str(extract_unique_values(row, coauthor_cols)), axis=1) if coauthor_cols else '[]'
df_clean['images'] = df_raw[image_cols].apply(lambda row: str([str(v).strip() for v in extract_unique_values(row, image_cols) if str(v).startswith('http')]), axis=1) if image_cols else '[]'

print(f"✅ Parsed all list fields")

# Parse child posts and comments
print("\n🔄 Parsing child posts and comments...")

child_post_cols = [col for col in df_raw.columns if col.startswith('childPosts/')]
if child_post_cols:
    def extract_child_posts(row):
        indices = set()
        for col in child_post_cols:
            try:
                idx = int(col.split('/')[1])
                indices.add(idx)
            except:
                continue
        children = []
        for idx in sorted(indices)[:10]:
            child_id = row.get(f'childPosts/{idx}/id')
            if pd.notna(child_id):
                children.append({
                    'id': child_id,
                    'type': row.get(f'childPosts/{idx}/type', ''),
                    'url': row.get(f'childPosts/{idx}/url', '')
                })
        return str(children) if children else '[]'
    df_clean['childPosts'] = df_raw.apply(extract_child_posts, axis=1)
else:
    df_clean['childPosts'] = '[]'

comment_cols = [col for col in df_raw.columns if col.startswith('latestComments/') and '/text' in col]
if comment_cols:
    def extract_comments(row):
        indices = set()
        for col in comment_cols:
            try:
                idx = int(col.split('/')[1])
                indices.add(idx)
            except:
                continue
        comments = []
        for idx in sorted(indices)[:5]:
            text = row.get(f'latestComments/{idx}/text')
            if pd.notna(text):
                comments.append({
                    'id': row.get(f'latestComments/{idx}/id', ''),
                    'text': str(text)[:200],
                    'ownerUsername': row.get(f'latestComments/{idx}/ownerUsername', '')
                })
        return str(comments) if comments else '[]'
    df_clean['latestComments'] = df_raw.apply(extract_comments, axis=1)
else:
    df_clean['latestComments'] = '[]'

print(f"✅ Parsed nested objects")

# Music info
print("\n🔄 Parsing music info...")
if 'musicInfo/song_name' in df_raw.columns or 'musicInfo/artist_name' in df_raw.columns:
    def extract_music(row):
        if pd.notna(row.get('musicInfo/artist_name')) or pd.notna(row.get('musicInfo/song_name')):
            return str({
                'artist_name': row.get('musicInfo/artist_name', ''),
                'song_name': row.get('musicInfo/song_name', ''),
                'uses_original_audio': row.get('musicInfo/uses_original_audio', False)
            })
        return ''
    df_clean['musicInfo'] = df_raw.apply(extract_music, axis=1)
else:
    df_clean['musicInfo'] = ''

# Timestamp
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce', utc=True)

# Calculated fields
print("\n🔄 Creating calculated fields...")
df_clean['likes'] = df_clean['likesCount']
df_clean['comments'] = df_clean['commentsCount']
df_clean['views'] = df_clean['videoViewCount']
df_clean['engagement'] = df_clean['likes'] + df_clean['comments']
df_clean['caption_text'] = df_clean['caption']
df_clean['caption_len'] = df_clean['caption_text'].str.len()

def safe_count(s):
    try:
        lst = ast.literal_eval(s)
        return len(lst) if isinstance(lst, list) else 0
    except:
        return 0

df_clean['n_hashtags'] = df_clean['hashtags'].apply(safe_count)
df_clean['n_mentions'] = df_clean['mentions'].apply(safe_count)

# Post type
product_type_lower = df_clean['productType'].str.lower().fillna('')
type_lower = df_clean['type'].str.lower().fillna('')

conditions = [
    product_type_lower.str.contains('clips|reel'),
    (type_lower.str.contains('video')) & (df_clean['videoDuration'] > 0) & (df_clean['videoDuration'] <= 90),
    type_lower.str.contains('video'),
    type_lower.str.contains('sidecar|carousel'),
    df_clean['videoUrl'].notna() & df_clean['videoUrl'].str.startswith('http')
]
choices = ['Reel', 'Reel', 'Video', 'Sidecar', 'Video']
df_clean['post_type'] = np.select(conditions, choices, default='Image')

print(f"✅ Calculated fields created")

# Final columns
final_columns = [
    'id', 'type', 'shortCode', 'caption', 'url',
    'likesCount', 'commentsCount', 'videoViewCount', 'videoPlayCount',
    'likes', 'comments', 'views', 'engagement',
    'ownerUsername', 'ownerFullName', 'ownerId',
    'displayUrl', 'videoUrl', 'videoDuration',
    'dimensionsHeight', 'dimensionsWidth',
    'hashtags', 'mentions', 'taggedUsers', 'coauthorProducers',
    'caption_text', 'caption_len', 'n_hashtags', 'n_mentions',
    'post_type', 'productType',
    'timestamp', 'locationName', 'locationId',
    'firstComment', 'latestComments', 'childPosts', 'images',
    'isSponsored', 'isPinned', 'isCommentsDisabled',
    'alt', 'inputUrl', 'musicInfo'
]

for col in final_columns:
    if col not in df_clean.columns:
        df_clean[col] = np.nan

df_final = df_clean[final_columns].copy()

# Save
df_final.to_csv('instagram_cleaned.csv', index=False)

print("\n" + "="*70)
print("✅ SUCCESS!")
print("="*70)
print(f"\n📊 Processed {len(df_final)} posts")
print(f"\n💾 Saved to: instagram_cleaned.csv")
print("\n🚀 Next: Run 'python dashboard.py' to launch dashboard")