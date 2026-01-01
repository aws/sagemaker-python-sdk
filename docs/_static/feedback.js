// Feedback functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add feedback component to the very bottom of each page
    const article = document.querySelector('article[role="main"]') || document.querySelector('.bd-article') || document.querySelector('main');
    if (article) {
        const feedbackHTML = `
            <div class="feedback-container">
                <div class="feedback-question">Was this page helpful?</div>
                <div class="feedback-buttons">
                    <button class="feedback-btn" data-feedback="yes">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M7.493 18.75c-.425 0-.82-.236-.975-.632A7.48 7.48 0 016 15.375c0-1.75.599-3.358 1.602-4.634.151-.192.373-.309.6-.397.473-.183.89-.514 1.212-.924a9.042 9.042 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75 2.25 2.25 0 012.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558-.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H14.23c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23h-.777zM2.331 10.977a11.969 11.969 0 00-.831 4.398 12 12 0 00.52 3.507c.26.85 1.084 1.368 1.973 1.368H4.9c.445 0 .72-.498.523-.898a8.963 8.963 0 01-.924-3.977c0-1.708.476-3.305 1.302-4.666.245-.403-.028-.959-.5-.959H4.25c-.832 0-1.612.453-1.918 1.227z"/>
                        </svg>
                        Yes
                    </button>
                    <button class="feedback-btn" data-feedback="no">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M15.73 5.25h1.035A7.465 7.465 0 0118 9.375a7.465 7.465 0 01-1.235 4.125h-.148c-.806 0-1.534.446-2.031 1.08a9.04 9.04 0 01-2.861 2.4c-.723.384-1.35.956-1.653 1.715a4.498 4.498 0 00-.322 1.672V21a.75.75 0 01-.75.75 2.25 2.25 0 01-2.25-2.25c0-1.152.26-2.243.723-3.218C7.74 15.724 7.366 15 6.748 15H3.622c-1.026 0-1.945-.694-2.054-1.715A12.134 12.134 0 011.5 12c0-2.848.992-5.464 2.649-7.521C4.537 3.997 5.136 3.75 5.754 3.75H9.77a4.5 4.5 0 011.423.23l3.114 1.04a4.5 4.5 0 001.423.23zM21.669 14.023c.536-1.362.831-2.845.831-4.398 0-1.22-.182-2.398-.52-3.507-.26-.85-1.084-1.368-1.973-1.368H19.1c-.445 0-.72.498-.523.898.591 1.2.924 2.55.924 3.977a8.958 8.958 0 01-1.302 4.666c-.245.403.028.959.5.959h1.053c.832 0 1.612-.453 1.918-1.227z"/>
                        </svg>
                        No
                    </button>
                </div>
                <div class="feedback-options">
                    <div class="feedback-options-title">What is the reason for your feedback? <span class="required">*</span></div>
                    <div class="feedback-checkboxes">
                        <label class="feedback-checkbox">
                            <input type="checkbox" data-reason="hard-to-understand">
                            <span class="checkmark"></span>
                            Content is hard to understand
                        </label>
                        <label class="feedback-checkbox">
                            <input type="checkbox" data-reason="code-doesnt-work">
                            <span class="checkmark"></span>
                            Procedure or code doesn't work
                        </label>
                        <label class="feedback-checkbox">
                            <input type="checkbox" data-reason="couldnt-find">
                            <span class="checkmark"></span>
                            Couldn't find what I need
                        </label>
                        <label class="feedback-checkbox">
                            <input type="checkbox" data-reason="out-of-date">
                            <span class="checkmark"></span>
                            Out of date/obsolete
                        </label>
                        <label class="feedback-checkbox">
                            <input type="checkbox" data-reason="other">
                            <span class="checkmark"></span>
                            Other
                        </label>
                    </div>
                    <div class="feedback-more">
                        <div class="feedback-more-title">Tell us more.</div>
                        <textarea class="feedback-textarea" placeholder="Please provide feedback on how we can improve this content. If applicable, provide the first part of the sentence or string at issue."></textarea>
                        <button class="feedback-submit-btn">Submit</button>
                    </div>
                </div>
                <div class="feedback-thanks">Thank you for your feedback!</div>
            </div>
        `;
        
        article.insertAdjacentHTML('beforeend', feedbackHTML);
        
        // Add click handlers
        const feedbackBtns = document.querySelectorAll('.feedback-btn');
        const thanksMessage = document.querySelector('.feedback-thanks');
        const feedbackOptions = document.querySelector('.feedback-options');
        const checkboxes = document.querySelectorAll('.feedback-checkbox input[type="checkbox"]');
        const submitBtn = document.querySelector('.feedback-submit-btn');
        const textarea = document.querySelector('.feedback-textarea');
        
        feedbackBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const feedback = this.dataset.feedback;
                
                // Remove selected class from all buttons
                feedbackBtns.forEach(b => b.classList.remove('selected'));
                
                // Add selected class to clicked button
                this.classList.add('selected');
                
                if (feedback === 'yes') {
                    // Hide options and show thanks
                    feedbackOptions.style.display = 'none';
                    thanksMessage.style.display = 'block';
                    
                    // Send positive feedback
                    if (typeof gtag !== 'undefined') {
                        gtag('event', 'page_feedback', {
                            'feedback_value': 'positive',
                            'page_location': window.location.href
                        });
                    }
                } else {
                    // Show options for negative feedback
                    feedbackOptions.style.display = 'block';
                    thanksMessage.style.display = 'none';
                }
            });
        });
        
        // Handle submit button
        submitBtn.addEventListener('click', function() {
            const selectedReasons = [];
            checkboxes.forEach(checkbox => {
                if (checkbox.checked) {
                    selectedReasons.push(checkbox.dataset.reason);
                }
            });
            const additionalFeedback = textarea.value.trim();
            
            // Hide options and show thanks
            feedbackOptions.style.display = 'none';
            thanksMessage.style.display = 'block';
            
            // Send negative feedback with details
            if (typeof gtag !== 'undefined') {
                gtag('event', 'page_feedback', {
                    'feedback_value': 'negative',
                    'feedback_reasons': selectedReasons,
                    'feedback_details': additionalFeedback,
                    'page_location': window.location.href
                });
            }
        });
    }
});
