const myImage = document.querySelector("img");

myImage.onclick = () => {
  const mySrc = myImage.getAttribute("src");
  if (mySrc === "4.jpg") {
    myImage.setAttribute("src", "5.jpg");
  } else {
    myImage.setAttribute("src", "4.jpg");
  }
};