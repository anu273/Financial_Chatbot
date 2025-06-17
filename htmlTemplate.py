css = '''
<style>
.chat-message {
    display: flex;
    margin-bottom: 1rem;
    align-items: flex-start;
    width: 100%;
}

.chat-message.bot {
    flex-direction: row;
    border-radius: 0.5rem;
    padding: 1rem;
}

.chat-message.user {
    flex-direction: row-reverse;
    justify-content: flex-end;
    background-color: #3e4451;
    border-radius: 0.5rem;
    padding: 1rem;
    width: 50%;
    margin-left: auto;
}

.chat-message .avatar { 
    width: 24px;
    height: 24px;
    flex-shrink: 0;
}

.chat-message .avatar img {
    width: 24px;
    height: 24px;
    border-radius: 50%; 
    object-fit: cover;
}

.chat-message .message {
    margin-left: 1rem;
    margin-right: 1rem;
    color: #fff;
    flex: 1;
    word-wrap: break-word;
}

body {
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: white;
    padding: 1rem;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    z-index: 999;
}

.chat-history-container {
    padding-bottom: 100px; /* Give space for the fixed input box */
    max-height: calc(100vh - 140px);
    overflow-y: auto;
}
</style>
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

