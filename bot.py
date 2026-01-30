import telebot
from gtts import gTTS
import os
import tempfile

# Bot token
BOT_TOKEN = "7742610579:AAGxsFcjBos_sE7G75-eAlg6EHbHfU1zMs8"

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🎤 *Text to Speech Bot*

Mujhe koi bhi text bhejo, main usko voice message mein convert kar dunga!

*Kaise use karein:*
- Bas apna text message bhejo
- Main aapko voice message return karunga

*Language support:*
- Hindi aur English dono supported hain
- Mixed text bhi kaam karega

Chalo shuru karte hain! 🚀
    """
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def text_to_speech(message):
    try:
        # User ka text
        text = message.text
        
        if not text or len(text.strip()) == 0:
            bot.reply_to(message, "Kripya kuch text bhejein!")
            return
        
        # Processing message
        processing_msg = bot.reply_to(message, "🔄 Voice bana raha hoon, thoda wait karein...")
        
        # Temporary file create karo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        # Text ko speech mein convert karo
        # Hindi text ke liye 'hi' language use karenge
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(temp_file)
        
        # Voice message bhejo
        with open(temp_file, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
        
        # Processing message delete karo
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
        # Temp file delete karo
        os.unlink(temp_file)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error aaya: {str(e)}\nKripya dubara try karein!")
        print(f"Error: {e}")

# Bot start karo
if __name__ == '__main__':
    print("Bot chal raha hai...")
    bot.infinity_polling()