let slideIndex=0;
const carousels=document.querySelector(".carousels");
console.log("Okay")

function showSlides(n){
  console.log(n)
    let slide=document.getElementsByClassName("slide");
    // console.log(slide[1]);
    for(let i=0;i<slide.length;i++){
        slide[i].style.display="none";
    }
    if(n>=slide.length){
        slideIndex=0;
    }
    if(n<0){
        slideIndex=slide.length-1;
    }
    slide[slideIndex].style.display="block";
    slide[slideIndex].style.animation="fade 1s";


}

document.getElementById('right').addEventListener('click',function(){
    showSlides(slideIndex+=1);
   carousels.style.transition = "all 0.3s ease-in-out";
   
    
})
document.getElementById('left').addEventListener('click',function(){
    showSlides(slideIndex-=1);
})

showSlides(slideIndex);