import streamlit as st
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_mistralai import ChatMistralAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    streaming=True,
)

# ── Node ──────────────────────────────────────────────────────────────────────

def chat_node(state: ChatState):
    messages = state["messages"][-6:]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

# ── Graph ─────────────────────────────────────────────────────────────────────

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ── Compile ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_chatbot():
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

chatbot = get_chatbot()

# =============================================================================
# ✅ CHANGE 1: All session_state keys needed for resume chat
#
# BEFORE: only "history" existed
#   if "history" not in st.session_state:
#       st.session_state.history = []
#
# AFTER: we now also track:
#   - "sessions"    → dict that stores chat history per thread_id
#                     { "Session 1": [...messages...], "Session 2": [...] }
#   - "active_id"   → which session is currently open
#   - "counter"     → used to generate unique session names
#
# WHY:
#   Resume chat means the user can switch between old conversations.
#   Each conversation needs its own thread_id (so LangGraph memory is separate)
#   and its own message history (so the UI shows the right messages).
# =============================================================================

if "sessions" not in st.session_state:
    st.session_state.sessions = {"Session 1": []}   # name → list of messages

if "active_id" not in st.session_state:
    st.session_state.active_id = "Session 1"        # currently open session

if "counter" not in st.session_state:
    st.session_state.counter = 1                    # used to name new sessions


# =============================================================================
# ✅ CHANGE 2: Helper — get messages for the active session
#
# BEFORE: st.session_state.history  (one flat list)
#
# AFTER:  st.session_state.sessions[active_id]
#         This looks up the message list for whichever session is open.
#
# WHY:
#   Instead of one global "history" list, we now have a dict of lists.
#   This helper just makes reading the active list cleaner.
# =============================================================================

def active_messages():
    return st.session_state.sessions[st.session_state.active_id]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💬 Sessions")

    # =========================================================================
    # ✅ CHANGE 3: "New Chat" button creates a new session
    #
    # BEFORE: no such button existed
    #
    # AFTER:
    #   Clicking this increments the counter, creates a new empty entry in
    #   st.session_state.sessions, and sets active_id to the new session.
    #
    # WHY:
    #   The user needs a way to start a fresh conversation while keeping
    #   old ones intact. Each new session gets a unique name like "Session 2",
    #   "Session 3", etc.
    # =========================================================================

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.counter += 1
        new_name = f"Session {st.session_state.counter}"
        st.session_state.sessions[new_name] = []
        st.session_state.active_id = new_name
        st.rerun()

    st.divider()

    # =========================================================================
    # ✅ CHANGE 4: Session list — clicking resumes a past chat
    #
    # BEFORE: no session list existed
    #
    # AFTER:
    #   We loop over all sessions and show a button for each one.
    #   Clicking a session button sets active_id to that session name,
    #   which causes the main area to load that session's messages.
    #
    # WHY:
    #   This is the actual "resume" mechanism.
    #   active_id acts like a pointer — changing it switches the conversation.
    #   LangGraph uses the session name as thread_id, so the LLM memory
    #   also switches automatically.
    # =========================================================================

    for session_name in reversed(list(st.session_state.sessions.keys())):
        is_active = session_name == st.session_state.active_id

        # Show a filled button for active session, plain for others
        label = f"▶ {session_name}" if is_active else f"   {session_name}"

        if st.button(label, key=f"btn_{session_name}", use_container_width=True):
            st.session_state.active_id = session_name
            st.rerun()

    # =========================================================================
    # ✅ CHANGE 5: Delete button — removes the active session
    #
    # BEFORE: no delete existed
    #
    # AFTER:
    #   Deletes the current session from the dict, then switches to the
    #   most recent remaining session. If all sessions are deleted,
    #   a fresh "Session 1" is created automatically.
    #
    # WHY:
    #   Without a delete option, old sessions accumulate forever.
    # =========================================================================

    st.divider()
    if st.button("🗑 Delete Current Chat", use_container_width=True):
        del st.session_state.sessions[st.session_state.active_id]
        if st.session_state.sessions:
            st.session_state.active_id = list(st.session_state.sessions.keys())[-1]
        else:
            st.session_state.sessions = {"Session 1": []}
            st.session_state.active_id = "Session 1"
            st.session_state.counter = 1
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("LangGraph Chatbot")

# =========================================================================
# ✅ CHANGE 6: Show the active session name as a subtitle
#
# BEFORE: nothing
#
# AFTER:  st.caption(f"Session: {st.session_state.active_id}")
#
# WHY:
#   Helps the user know which session they are currently in.
# =========================================================================

st.caption(f"Session: {st.session_state.active_id}")

# Render messages for the active session
for msg in active_messages():
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type a message...")

if user_input:
    active_messages().append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("ai"):

        def token_generator():
            for chunk in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                # =====================================================================
                # ✅ CHANGE 7: thread_id is now the active session name, not hardcoded
                #
                # BEFORE:
                #   config={"configurable": {"thread_id": "user_1"}}
                #
                # AFTER:
                #   config={"configurable": {"thread_id": st.session_state.active_id}}
                #
                # WHY:
                #   thread_id is how LangGraph separates memory between conversations.
                #   "user_1" was hardcoded — every session shared the same memory.
                #   Now each session has its own thread_id, so when you resume
                #   "Session 2", LangGraph loads that session's memory, not Session 1's.
                # =====================================================================
                config={"configurable": {"thread_id": st.session_state.active_id}},
                stream_mode="messages",
            ):
                message_chunk, metadata = chunk
                if message_chunk.content:
                    yield message_chunk.content

        bot_reply = st.write_stream(token_generator())

    active_messages().append({"role": "ai", "content": bot_reply})



    # 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