const CACHE_NAME = 'michibiki-pwa-v1';
const urlsToCache = [
  '/',
  '/login',
  '/static/style.css',
  '/static/manifest.json',
  '/static/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});
