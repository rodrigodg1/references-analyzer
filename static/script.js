document.addEventListener('DOMContentLoaded', function () {
    const citationLinks = document.querySelectorAll('.citation-link');
    const citationInfoArea = document.getElementById('citation-info-area');
    const citationDetailsDiv = document.getElementById('citation-details');

    citationLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default link navigation

            const citationKey = this.dataset.citationKey;
            // **URL-encode the citationKey:**
            const encodedCitationKey = encodeURIComponent(citationKey);
            const citationDetailURL = `/get_citation_detail/${encodedCitationKey}`; // Construct URL for Flask route

            fetch(citationDetailURL) // Fetch citation details from Flask server
                .then(response => response.json())
                .then(data => {
                    if (data.citation_info) {
                        let detailsHTML = "<ul>"; // Start unordered list

                        if (data.citation_info.doi) {
                            detailsHTML += `<li><strong>DOI:</strong> ${data.citation_info.doi}</li>`;
                        }

                        if (data.citation_info.title) {
                            detailsHTML += `<li><strong>Title:</strong> ${data.citation_info.title}</li>`;
                        }
                        if (data.citation_info.author) {
                            detailsHTML += `<li><strong>Author(s):</strong> ${data.citation_info.author}</li>`;
                        }
                        if (data.citation_info.year) {
                            detailsHTML += `<li><strong>Year:</strong> ${data.citation_info.year}</li>`;
                        }
                        if (data.citation_info.journal) {
                            detailsHTML += `<li><strong>Journal:</strong> ${data.citation_info.journal}</li>`;
                        }
                        detailsHTML += "</ul>"; // Close unordered list

                        citationDetailsDiv.innerHTML = detailsHTML;
                        citationInfoArea.style.display = 'block'; // Show citation info area
                    } else if (data.error) {
                        citationDetailsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                        citationInfoArea.style.display = 'block';
                    } else {
                        citationDetailsDiv.innerHTML = `<p>Citation details not found for key: ${citationKey}</p>`;
                        citationInfoArea.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Error fetching citation info:', error);
                    citationDetailsDiv.innerHTML = `<p>Error fetching citation information.</p>`;
                    citationInfoArea.style.display = 'block';
                });
        });
    });

});