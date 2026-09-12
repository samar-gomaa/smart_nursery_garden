from telegram import Bot
import asyncio

BOT_TOKEN = "8974053094:AAFYLitloDgBX3sA-C2RAuvw3Co4mNNh5dI"
CHAT_ID = "1841609148"


def send_gas_alert():
    message = " WARNING!\nGas or smoke detected in the nursery."

    async def send_message():
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message
        )

    asyncio.run(send_message())


# Test
if __name__ == "__main__":
    send_gas_alert()