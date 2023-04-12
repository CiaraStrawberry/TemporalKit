function switch_to_temporal_kit() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[6].click();
   // gradioApp().getElementById('TemporalKit').querySelectorAll('button')[0].click();
}

function switch_to_temporal_kit_final2() {
    // Get the current image source
 //   const gallery = document.getElementById('img2img_gallery');
//    const firstImageSource = gallery.getElementsByTagName('img')[0].src;
    //firstImageSource = document.getElementById('svelte-1tkea93').src;
    // Switch to the "temporal-kit" tab and the "final" subtab

    switch_to_temporal_kit_final();
    const tabList = document.querySelector("#TemporalKit-Tab");

    if (tabList) {
        const firstChild = tabList.firstElementChild;
        
        if (firstChild) {
            const secondTab = firstChild.querySelector(":nth-child(2)");
            
            if (secondTab) {
                secondTab.click();
            } else {
                console.error("Second tab element not found.");
            }
        } else {
            console.error("First child element not found.");
        }
    } else {
        console.error("Tab list element not found.");
    }
    document.getElementById("read_last_settings").click();
    document.getElementById("read_last_image").click();

    // Paste the image source into the "final" subtab's image element
 //   document.getElementById('output_image').src = firstImageSource;
}

