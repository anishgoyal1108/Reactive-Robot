/**
 * Reactive Robot - Interactive JavaScript
 * Handles navigation toggle and smooth scrolling
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const navToggle = document.getElementById("navToggle");
  const navMenu = document.getElementById("navMenu");
  const navbar = document.getElementById("navbar");
  const navLinks = document.querySelectorAll(".nav-menu a");

  // Mobile Navigation Toggle
  navToggle?.addEventListener("click", () => {
    navMenu.classList.toggle("active");
    const icon = navToggle.querySelector(".material-icons");
    icon.textContent = navMenu.classList.contains("active") ? "close" : "menu";
  });

  // Close mobile menu when clicking a link
  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      navMenu.classList.remove("active");
      const icon = navToggle?.querySelector(".material-icons");
      if (icon) icon.textContent = "menu";
    });
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      e.preventDefault();
      const target = document.querySelector(anchor.getAttribute("href"));
      if (target) {
        const navHeight = navbar?.offsetHeight || 0;
        const targetPosition =
          target.getBoundingClientRect().top + window.pageYOffset - navHeight;

        window.scrollTo({
          top: targetPosition,
          behavior: "smooth",
        });
      }
    });
  });

  // Navbar shadow on scroll
  let lastScroll = 0;
  window.addEventListener(
    "scroll",
    () => {
      const currentScroll = window.pageYOffset;

      if (currentScroll > 100) {
        navbar?.classList.add("scrolled");
      } else {
        navbar?.classList.remove("scrolled");
      }

      lastScroll = currentScroll;
    },
    { passive: true },
  );

  // Intersection Observer for scroll animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  }, observerOptions);

  // Observe elements for animation
  document
    .querySelectorAll(".feature-card, .stat-card, .timeline-item, .team-card")
    .forEach((el) => {
      el.style.opacity = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
      observer.observe(el);
    });

  // Add visible class styles dynamically
  const style = document.createElement("style");
  style.textContent = `
        .feature-card.visible,
        .stat-card.visible,
        .timeline-item.visible,
        .team-card.visible {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    `;
  document.head.appendChild(style);

  // Console welcome message
  console.log(
    "%c🤖 Reactive Robot",
    "font-size: 24px; font-weight: bold; color: #1976D2;",
  );
  console.log(
    "%cHuman-Aware Trajectory Replanning",
    "font-size: 14px; color: #757575;",
  );
});
