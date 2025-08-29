import streamlit as st

from dotenv import load_dotenv

# ------------------------------------------------------------
# Expert LLM Helper（Streamlit × LangChain）
# 条件対応：
# ① 単一入力フォーム → LLM応答を画面表示
# ② ラジオボタンで専門家の種類を切替（システムメッセージ変更）
# ③ (入力テキスト, 選択値) -> 回答 を返す関数 query_llm を定義・利用
# ④ 使い方テキストの明示
# ⑤ デプロイは Python 3.11 を使用（Cloud 側設定）
# ------------------------------------------------------------
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====== OpenAI API Key（環境変数→Secretsの順に取得。Secretsは例外で包む） ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Streamlit Cloud で設定
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.set_page_config(page_title="Expert LLM Helper", page_icon="🤖", layout="centered")
    st.error(
        "OPENAI_API_KEY が未設定です。\n"
        "・ローカル/VSCode: ターミナルで `set OPENAI_API_KEY=sk-...`（一時）または `setx OPENAI_API_KEY \"sk-...\"`（永続）\n"
        "・Streamlit Cloud: App settings → Secrets に `OPENAI_API_KEY = \"sk-...\"` を保存\n"
    )
    st.stop()

# ====== LLM 準備（キャッシュ） ==============================================
@st.cache_resource(show_spinner=False)
def get_llm():
    # 必要に応じて "gpt-4o" 等に変更可
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3,
        timeout=60,
    )

# ====== 専門家ペルソナ（自由に追加/編集OK） ================================
PERSONAS = {
    "コンプライアンス（保険業）": (
        "あなたは生命保険業界のコンプライアンス専門家です。"
        "保険業法・当局ガイドライン・社内規程の観点から、"
        "潜在リスク・必要な社内手続・留意点を、根拠を示しつつ簡潔に助言してください。"
        "不確実な点は推測せず、確認すべき一次情報と部署を提示してください。"
    ),
    "データサイエンス": (
        "あなたはビジネス課題を分析で解くデータサイエンス専門家です。"
        "目的定義→データ要件→前処理→特徴量→モデル選定→評価指標→運用化の順で、"
        "現場で再現可能なステップを具体的に提案してください。"
    ),
    "CX/サービスデザイン": (
        "あなたはCX/サービスデザインの専門家です。"
        "ペルソナ・ジャーニー・価値仮説・MVP・検証方法・KPIを整理し、"
        "実装優先度（High/Medium/Low）付きで改善案を提示してください。"
    ),
}

# ====== 条件③：コア関数（入力テキスト＋選択値 → 回答文字列） ==============
def query_llm(input_text: str, persona_key: str) -> str:
    """
    Args:
        input_text: 画面の入力テキスト
        persona_key: ラジオで選択した専門家の種類（PERSONASのキー）
    Returns:
        LLMの回答テキスト
    """
    system_msg = PERSONAS.get(
        persona_key,
        "あなたは有能なアシスタントです。簡潔かつ具体的に答えてください。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{user_input}"),
        ]
    )
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"user_input": input_text})

# ====== UI 構成 =============================================================
st.set_page_config(page_title="Expert LLM Helper", page_icon="🤖", layout="centered")
st.title("🤖 Expert LLM Helper")

# 概要・操作方法（条件④）
with st.expander("ℹ️ アプリ概要・操作方法（必読）", expanded=True):
    st.markdown(
        """
**できること**  
- テキスト入力と**専門家の種類**の選択に基づき、LLMが専門家視点で回答します。

**使い方**  
1. 左（または上）のラジオで **専門家の種類** を選択  
2. 下のテキスト欄に相談内容を入力  
3. **送信** をクリック → 回答が表示されます

**注意**  
- 本出力は支援情報です。重要事項は一次情報で必ず確認し、関係部署レビューを受けてください。
        """
    )

# ラジオ（条件②）
persona = st.radio("専門家の種類を選択してください：", list(PERSONAS.keys()), horizontal=True)

# 入力フォーム（条件①）
with st.form("input_form", clear_on_submit=False):
    user_text = st.text_area(
        "入力テキスト",
        placeholder="例）新商品の広告表現の留意点／分析の進め方／顧客体験の改善アイデア など",
        height=160,
    )
    submitted = st.form_submit_button("送信")

# 実行
if submitted:
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中…"):
            try:
                answer = query_llm(user_text, persona)
                st.markdown("### 📝 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")





