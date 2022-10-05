import pywhatkit

# Сообщение на телефон в what's app

# phone_number = input("Enter phone number: ")
# Отправить сообщение в 17:56 а затем закрыть окно через 2 секунды
# pywhatkit.sendwhatmsg(phone_number, "Test", 17, 56, 15, True, 2)

# Сообщение группе в what's app

group_id = input("Enter group id: ")
pywhatkit.sendwhatmsg_to_group(group_id, "Test Group", 18,15)
