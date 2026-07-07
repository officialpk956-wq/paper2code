import http from 'k6/http';
import { check, sleep, group } from 'k6';

export const options = {
  vus: 10,
  duration: '15m',
  thresholds: {
    http_req_duration: ['p(95)<100', 'p(99)<200'],
    http_req_failed: ['rate<0.01'],
  },
  stages: [
    { duration: '1m', target: 10 },    // Ramp up
    { duration: '10m', target: 10 },   // Sustain
    { duration: '4m', target: 0 },     // Ramp down
  ],
};

const BASE_URL = __ENV.BASE_URL || 'https://api.paper2code.com';

export default function () {
  group('GET /problems', () => {
    let res = http.get(`${BASE_URL}/api/problems`);
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 100ms': (r) => r.timings.duration < 100,
    });
  });

  sleep(0.5);

  group('GET /papers', () => {
    let res = http.get(`${BASE_URL}/api/papers`);
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 200ms': (r) => r.timings.duration < 200,
    });
  });

  sleep(0.5);

  group('GET /analytics/dashboard', () => {
    let res = http.get(`${BASE_URL}/api/analytics/dashboard`, {
      headers: {
        'X-Learner-ID': `anon-${__VU}-${__ITER}`,
      },
    });
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 300ms': (r) => r.timings.duration < 300,
    });
  });

  sleep(1);
}
