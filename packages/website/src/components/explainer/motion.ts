import { useEffect, useState } from 'react';

export function usePresentationMotion() {
  const [requested, setRequested] = useState(true);
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const preference = matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(preference.matches);
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);
  const enabled = requested && !reduced;
  useEffect(() => {
    const elements = [...document.querySelectorAll<HTMLElement>('[data-parallax]')];
    const reveals = [...document.querySelectorAll<HTMLElement>('.reveal')];
    let raf = 0;
    const update = () => {
      raf = 0;
      const height = innerHeight;
      for (const element of elements) {
        const rect = element.getBoundingClientRect();
        const depth = Number(element.dataset.parallax ?? 0.1);
        const shift = enabled ? Math.max(-55, Math.min(55, (height / 2 - rect.top - rect.height / 2) * depth)) : 0;
        element.style.setProperty('--parallax-y', `${shift}px`);
      }
    };
    const scroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries)
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
      },
      { threshold: 0.08 },
    );
    for (const element of reveals) observer.observe(element);
    document.documentElement.dataset.presentationMotion = enabled ? 'on' : 'off';
    update();
    window.addEventListener('scroll', scroll, { passive: true });
    window.addEventListener('resize', scroll);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('scroll', scroll);
      window.removeEventListener('resize', scroll);
      delete document.documentElement.dataset.presentationMotion;
    };
  }, [enabled]);
  return { enabled, reduced, toggle: () => setRequested((value) => !value) };
}
