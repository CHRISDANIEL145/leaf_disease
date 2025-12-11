/**
 * LeafAI - Main JavaScript
 * Futuristic UI Interactions and Animations
 */

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    initParallax();
    initScrollAnimations();
    initSmoothScroll();
    initTypingEffect();
    initTooltips();
    initMobileMenu();
});

/**
 * Parallax Effect for Background Elements
 */
function initParallax() {
    const parallaxElements = document.querySelectorAll('[data-parallax]');

    if (parallaxElements.length === 0) return;

    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;

        parallaxElements.forEach(el => {
            const speed = parseFloat(el.dataset.parallax) || 0.5;
            const yPos = -(scrolled * speed);
            el.style.transform = `translate3d(0, ${yPos}px, 0)`;
        });
    });
}

/**
 * Scroll-triggered Animations
 */
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.animate-on-scroll');

    if (animatedElements.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    animatedElements.forEach(el => observer.observe(el));
}

/**
 * Smooth Scrolling for Anchor Links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            e.preventDefault();
            const target = document.querySelector(targetId);

            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Typing Effect for Hero Text
 */
function initTypingEffect() {
    const typingElements = document.querySelectorAll('[data-typing]');

    typingElements.forEach(el => {
        const text = el.dataset.typing;
        let index = 0;
        el.textContent = '';

        const typeChar = () => {
            if (index < text.length) {
                el.textContent += text.charAt(index);
                index++;
                setTimeout(typeChar, 100);
            }
        };

        // Start typing when element is visible
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                typeChar();
                observer.disconnect();
            }
        });

        observer.observe(el);
    });
}

/**
 * Tooltip Initialization
 */
function initTooltips() {
    // Tooltips are handled via CSS, but we can add dynamic ones here
}

/**
 * Mobile Menu Handler
 */
function initMobileMenu() {
    const menuBtn = document.getElementById('mobile-menu-btn');
    const menu = document.getElementById('mobile-menu');

    if (!menuBtn || !menu) return;

    menuBtn.addEventListener('click', () => {
        menu.classList.toggle('hidden');

        // Update icon
        const icon = menuBtn.querySelector('i');
        if (menu.classList.contains('hidden')) {
            icon.setAttribute('data-lucide', 'menu');
        } else {
            icon.setAttribute('data-lucide', 'x');
        }
        lucide.createIcons();
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!menuBtn.contains(e.target) && !menu.contains(e.target)) {
            menu.classList.add('hidden');
        }
    });
}

/**
 * Circular Progress Animation
 */
function animateCircularProgress(element, targetValue) {
    let currentValue = 0;
    const duration = 1500;
    const step = targetValue / (duration / 16);

    const animate = () => {
        currentValue = Math.min(currentValue + step, targetValue);
        element.style.setProperty('--progress', `${currentValue}%`);

        const valueDisplay = element.querySelector('span');
        if (valueDisplay) {
            valueDisplay.textContent = `${Math.round(currentValue)}%`;
        }

        if (currentValue < targetValue) {
            requestAnimationFrame(animate);
        }
    };

    animate();
}

/**
 * Number Counter Animation
 */
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const startTime = performance.now();

    const updateCounter = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = Math.round(start + (target - start) * easeOutQuart);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    };

    requestAnimationFrame(updateCounter);
}

/**
 * Toast Notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-6 right-6 px-6 py-4 rounded-xl z-50 flex items-center gap-3 animate-slide-up ${type === 'success' ? 'bg-neon-green/20 border border-neon-green/30 text-neon-green' :
            type === 'error' ? 'bg-red-500/20 border border-red-500/30 text-red-400' :
                'bg-neon-teal/20 border border-neon-teal/30 text-neon-teal'
        }`;

    toast.innerHTML = `
        <i data-lucide="${type === 'success' ? 'check-circle' : type === 'error' ? 'alert-circle' : 'info'}" class="w-5 h-5"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(toast);
    lucide.createIcons();

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Loading Overlay
 */
function showLoading(message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'fixed inset-0 z-50 bg-space-900/90 backdrop-blur-sm flex items-center justify-center';
    overlay.innerHTML = `
        <div class="text-center">
            <div class="w-16 h-16 border-4 border-neon-teal border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-neon-teal font-medium">${message}</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Format Confidence Percentage
 */
function formatConfidence(value) {
    return `${value.toFixed(1)}%`;
}

/**
 * Debounce Function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Add slide-up animation keyframe
 */
const style = document.createElement('style');
style.textContent = `
    @keyframes slide-up {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .animate-slide-up {
        animation: slide-up 0.3s ease-out forwards;
    }
`;
document.head.appendChild(style);

// Export functions for global use
window.LeafAI = {
    showToast,
    showLoading,
    hideLoading,
    animateCircularProgress,
    animateCounter,
    formatConfidence
};
