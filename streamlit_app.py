import streamlit as st
name = st.text_input("Enter Your name", "Type Here ...")
result = name.title()
st.success(result)

status = st.radio("Select Gender: ", ('Male', 'Female'))

if (status == 'Male'):
    st.success("Male")
else:
    st.success("Female")

 hobby = st.selectbox("Hobbies: ",
                     ['Dancing', 'Reading', 'Sports'])
 st.write("Your hobby is: ", hobby)


