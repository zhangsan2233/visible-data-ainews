import sqlite3
import json
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import math
from opencc import OpenCC
from urllib.parse import urlparse
import re
from collections import Counter
import os
from openai import OpenAI

# === 配置参数 ===
DB_NAME = "sample.db"
OUTPUT_JSON = "news_graph_causal.json"
DEEPSEEK_API_KEY = "sk-f0f1640d654748f69a61d4eec3ec9192"  # 替换为你的DeepSeek API Key

SHARED_CATEGORY_WEIGHT = 0.2
SHARED_ENTITY_WEIGHT = 0.6
TIME_DECAY_LAMBDA = 3
MIN_WEIGHT = 0.55
SAME_DOMAIN_WEIGHT = 0.05

# === 初始化DeepSeek客户端 ===
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# === 简繁转换器 ===
cc = OpenCC('t2s')

print("📖 Loading news from DB...")

# === 从数据库加载新闻，包括 URL ===
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("""
    SELECT id, title, description, publishedAt, categories, url
    FROM ai_news
    WHERE title IS NOT NULL AND description IS NOT NULL
""")
rows = cursor.fetchall()
conn.close()

print(f"✅ Loaded {len(rows)} news records.")

# === 构造节点 ===
nodes = []
news_data = []
for nid, title, desc, pub, cats, url in rows:
    title_s = cc.convert(title.strip())
    desc_s = cc.convert(desc.strip())
    pub_date = pub[:10] if pub else "unknown"
    categories_s = [cc.convert(c.strip()) for c in cats.split(",")] if cats else []

    news_data.append({
        "id": f"news_{nid}",
        "title": title_s,
        "summary": desc_s,
        "time": pub_date,
        "categories": categories_s,
        "url": url
    })
    nodes.append({
        "id": f"news_{nid}",
        "type": "news",
        "title": title_s,
        "summary": desc_s[:1000],
        "time": pub_date,
        "url": url
    })

# === 文本向量模型 ===
print("🧠 Generating embeddings for title + summary...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [n["title"] + ". " + n["summary"] for n in news_data]
embeddings = model.encode(texts)

# === 实体提取 ===
print("🔍 Extracting named entities...")
nlp = spacy.load("zh_core_web_sm")
entities_list = []
for n in news_data:
    doc = nlp(n["title"] + ". " + n["summary"])
    entities_list.append(set([ent.text for ent in doc.ents]))

# === 构造边 ===
print("🔗 Generating news-news relationships...")
links = []
for i in range(len(news_data)):
    for j in range(i + 1, len(news_data)):
        n1, n2 = news_data[i], news_data[j]

        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

        # 类别共享（对数缩放）
        cat_overlap = len(set(n1["categories"]) & set(n2["categories"]))
        cat_score = SHARED_CATEGORY_WEIGHT * math.log1p(cat_overlap)

        # 实体共享（Jaccard）
        ent_i, ent_j = entities_list[i], entities_list[j]
        if ent_i or ent_j:
            entity_jaccard = len(ent_i & ent_j) / len(ent_i | ent_j)
        else:
            entity_jaccard = 0
        entity_score = SHARED_ENTITY_WEIGHT * (entity_jaccard * 2.5)

        # 时间衰减
        time_decay = 1.0
        try:
            d1 = datetime.date.fromisoformat(n1["time"])
            d2 = datetime.date.fromisoformat(n2["time"])
            days_diff = (d2 - d1).days
            if days_diff < 0:
                continue
            time_decay = math.exp(-abs(days_diff) / TIME_DECAY_LAMBDA)
        except:
            pass

        # 同域名加权
        domain_bonus = 0
        try:
            domain1 = urlparse(n1["url"]).netloc
            domain2 = urlparse(n2["url"]).netloc
            if domain1 and domain1 == domain2:
                domain_bonus = SAME_DOMAIN_WEIGHT
        except:
            pass

        # 综合权重
        weight = (
            0.6 * sim +
            0.25 * entity_score +
            0.1 * cat_score
        ) * time_decay + domain_bonus

        if weight >= MIN_WEIGHT:
            links.append({
                "source": n1["id"],
                "target": n2["id"],
                "weight": round(weight, 2)
            })

print(f"✅ Generated {len(links)} refined news-news relationships.")

# === 检测连通图（簇）===
print("🔍 Detecting connected components...")

def find_connected_components(nodes, links):
    """检测图中的连通分量"""
    graph = {}
    for node in nodes:
        graph[node["id"]] = []
    
    for link in links:
        graph[link["source"]].append(link["target"])
        graph[link["target"]].append(link["source"])
    
    visited = set()
    components = []
    
    for node_id in graph:
        if node_id not in visited:
            # BFS遍历连通分量
            component = []
            queue = [node_id]
            visited.add(node_id)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(component) > 1:  # 只保留有连接的簇
                components.append(component)
    
    return components

# 检测连通分量
components = find_connected_components(nodes, links)
print(f"📊 Found {len(components)} connected components")

# === DeepSeek API 调用函数 ===
def call_deepseek_api(prompt, max_retries=3):
    """使用OpenAI SDK调用DeepSeek API"""
    for attempt in range(max_retries):
        try:
            print(f"  🔄 调用DeepSeek API (尝试 {attempt + 1}/{max_retries})...")
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的AI新闻编辑，擅长用简洁准确的语言概括新闻主题。请用中文回答，保持专业性和准确性。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=150,
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理输出
            content = re.sub(r'^["\']|["\']$', '', content)
            content = re.sub(r'\s+', ' ', content)
            
            print(f"  ✅ API调用成功")
            return content
            
        except Exception as e:
            print(f"  ⚠️ API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
            else:
                return None
    
    return None

def generate_cluster_summary_with_deepseek(component_nodes, news_data, cluster_idx):
    """使用DeepSeek生成簇概括"""
    # 收集簇内新闻信息
    cluster_titles = []
    
    for node_id in component_nodes:
        node = next((n for n in news_data if n["id"] == node_id), None)
        if node:
            cluster_titles.append(node["title"])
    
    if not cluster_titles:
        return f"AI新闻集群 {cluster_idx}"
    
    # 构建优化的提示词
    prompt = f"""请分析以下一组相关的AI新闻标题，用15-25字概括核心主题和技术焦点：

{chr(10).join([f"• {title}" for title in cluster_titles[:6]])}

请生成一个专业、简洁的中文概括，要求：
1. 准确反映核心主题和技术方向
2. 15-25字长度
3. 突出具体技术名称和关键应用
4. 避免通用描述

概括："""
    
    summary = call_deepseek_api(prompt)
    
    if summary and len(summary) >= 10 and len(summary) <= 50:
        return summary
    else:
        # 使用备用方案
        return generate_fallback_summary(cluster_titles, cluster_idx)

def generate_fallback_summary(titles, cluster_idx):
    """备用概括生成方案"""
    all_text = " ".join(titles)
    
    # 提取关键词
    words = re.findall(r'[\u4e00-\u9fff]+|[A-Z][a-z]*[A-Z][a-zA-Z]*', all_text)
    
    # 过滤停用词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会'}
    words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # 统计词频
    word_freq = Counter(words)
    top_keywords = [word for word, freq in word_freq.most_common(3)]
    
    # 识别技术术语
    tech_terms = list(set(re.findall(r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b', all_text)))
    
    if tech_terms and top_keywords:
        return f"{tech_terms[0]}技术：{', '.join(top_keywords[:2])}"
    elif tech_terms:
        return f"{tech_terms[0]}技术创新与发展"
    elif top_keywords:
        return f"AI{top_keywords[0]}领域进展与应用"
    else:
        return f"AI技术前沿集群 {cluster_idx}"

def generate_global_summary_with_deepseek(news_data):
    """使用DeepSeek生成全局摘要"""
    sample_titles = [n["title"] for n in news_data[:12]]
    
    prompt = f"""基于以下AI新闻标题，用20-30字概括当前AI领域的主要趋势和发展动态：

{chr(10).join([f"• {title}" for title in sample_titles])}

请生成一个全面、专业的中文概括，描述技术发展趋势和行业热点："""
    
    summary = call_deepseek_api(prompt, max_retries=2)
    
    if summary:
        return summary
    else:
        # 备用全局摘要
        return "AI技术快速发展，大模型、算力优化与创新应用成为行业焦点，各领域迎来技术突破"

# === 为每个连通图生成DeepSeek概括 ===
print("🧩 Generating cluster summaries with DeepSeek API...")
cluster_summaries = {}

# 由于有41个簇，我们只处理前20个以避免过多API调用
max_clusters_to_process = min(20, len(components))
successful_api_calls = 0

for idx, component in enumerate(components):
    if len(component) < 2:
        continue
        
    if idx >= max_clusters_to_process:
        # 对于超过限制的簇，使用简单命名
        cluster_id = f"cluster_{idx}"
        cluster_summaries[cluster_id] = {
            "summary": f"AI新闻集群 {idx}",
            "node_ids": component,
            "size": len(component)
        }
        continue
        
    print(f"  📍 处理簇 {idx} ({len(component)} 个节点)...")
    
    # 使用DeepSeek生成概括
    summary = generate_cluster_summary_with_deepseek(component, news_data, idx)
    cluster_id = f"cluster_{idx}"
    cluster_summaries[cluster_id] = {
        "summary": summary,
        "node_ids": component,
        "size": len(component)
    }
    
    # 统计成功调用
    if not any(word in summary for word in ["集群", "集群"]):
        successful_api_calls += 1
    
    print(f"  ✅ 簇 {idx}: {summary}")
    
    # 添加延迟避免API限制
    import time
    time.sleep(1)

# === 全局语义摘要 ===
print("🧩 Generating global semantic summary with DeepSeek API...")
semantic_summary = generate_global_summary_with_deepseek(news_data)

# === 添加 summary 节点 ===
summary_node = {
    "id": "summary_global",
    "type": "summary",
    "title": "新闻语义概括",
    "summary": semantic_summary,
    "time": "N/A",
    "url": "",
    "color": "#FFD700"
}
nodes.append(summary_node)

# === 连接 summary 节点到前 8 条新闻节点 ===
for n in news_data[:8]:
    links.append({
        "source": "summary_global",
        "target": n["id"],
        "weight": 0.3
    })

# === 输出 JSON，包含簇概括信息 ===
graph_data = {
    "force_graph": {"nodes": nodes, "links": links},
    "semantic_summary": semantic_summary,
    "cluster_summaries": cluster_summaries
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=2)

print(f"🎯 Graph with DeepSeek-generated cluster summaries saved to {OUTPUT_JSON}")
print(f"📊 Cluster info: {len(cluster_summaries)} clusters with summaries")
print(f"🔗 API Success: {successful_api_calls}/{max_clusters_to_process} clusters used DeepSeek API")
print(f"🧠 Global Summary: {semantic_summary}")