import pickle
import streamlit as st


model=pickle.load(open("spam.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))

def main():
	st.title("mail predictor")
	st.subheader("using python and streamlit")
	msg=st.text_input("enter mail:")
	if st.button("predict"):
		data=[msg]
		vecto=cv.transform(data).toarray()
		prediction=model.predict(vecto)
		result=prediction[0]
		if result==1:
			st.error("mail is spam")
		else:
			st.success("mail is ham")
main()