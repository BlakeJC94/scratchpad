// const myHeading = document.querySelector("h1");
// myHeading.textContent = "Foo!";

// Lol comment
// alert("hello!");

function mult(a, b) {
    let result = a * b
    return result
}


function poke() {
    alert("Boop!")
}

// document.querySelector("html").addEventListener("click", poke)


const myImage = document.querySelector("img")
myImage.onclick = () => {
    const mySrc = myImage.getAttribute("src")
    if (mySrc == "images/firefox-icon.png") {
        myImage.setAttribute("src", "images/firefox-icon2.png")
    } else {
        myImage.setAttribute("src", "images/firefox-icon.png")
    }
}


let myButton = document.querySelector("button")
let myHeading = document.querySelector("h1")

function setUserName() {
    const myName = prompt("Please enter name")
    if (!myName) {
        setUSerName()
    } else {
        localStorage.setItem("name", myName)
        myHeading.textContent = `Foo ${myName}`
    }
}

if (!localStorage.getItem("name")) {
    setUserName()
} else {
    const storedName = localStorage.getItem("name")
    myHeading.textContent = `Foo ${myName}`
}

myButton.onclick = () => {
    setUserName()
}
