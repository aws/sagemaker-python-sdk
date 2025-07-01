// _static/js/move_version_selector.js
document.addEventListener("DOMContentLoaded", function () {
    // Wait for the RTD version selector to load
    const interval = setInterval(function () {
        const footerVersion = document.querySelector("#rtd-footer-container");
        const navbarPlaceholder = document.querySelector("#rtd-version-selector");

        if (footerVersion && navbarPlaceholder) {
            // Move the child nodes (usually an iframe or dropdown) to the navbar
            while (footerVersion.firstChild) {
                navbarPlaceholder.appendChild(footerVersion.firstChild);
            }
            footerVersion.remove();  // optional: remove footer container
            clearInterval(interval);
        }
    }, 300);
});
