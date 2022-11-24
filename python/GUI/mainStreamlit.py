import streamlit as st

st.write("""# hello  world!""")
thing = st.expander("things",False)
def Flip():
  [theme]
  primaryColor="#F63366"
  backgroundColor="#FFFFFF"
with thing:
  st.write("""# why did  you open this?""")
  st.checkbox("are you alive?",True,on_change=Flip())
