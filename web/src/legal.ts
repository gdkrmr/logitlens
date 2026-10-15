import '@fontsource-variable/martian-mono/wdth.css';
import '@fontsource-variable/atkinson-hyperlegible-next/wght.css';
import './style.css';

// contact details are stored reversed and base64-encoded to keep them out of the plain HTML
document.querySelectorAll<HTMLElement>('[data-obf]').forEach((el) => {
  const text = atob(el.dataset.obf!).split('').reverse().join('');
  el.textContent = text;
  if (el instanceof HTMLAnchorElement) el.href = `mailto:${text}`;
});
