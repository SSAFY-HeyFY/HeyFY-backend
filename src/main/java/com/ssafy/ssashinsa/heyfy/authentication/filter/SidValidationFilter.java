package com.ssafy.ssashinsa.heyfy.authentication.filter;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.authentication.config.ApiPaths;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode; // ✨ ErrorCode를 임포트
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.common.util.RedisUtil;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

// SID 유효성만 검사하는 필터
@RequiredArgsConstructor
public class SidValidationFilter extends OncePerRequestFilter {

    private final RedisUtil redisUtil;
    private final AntPathMatcher pathMatcher = new AntPathMatcher();

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) throws ServletException {
        String requestURI = request.getRequestURI();
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        boolean isPublicPath = ApiPaths.PUBLIC_PATHS.stream()
                .anyMatch(pattern -> pathMatcher.match(pattern, requestURI));

        boolean isNonSensitivePath = ApiPaths.NON_SENSITIVE_PATHS.stream()
                .anyMatch(pattern -> pathMatcher.match(pattern, requestURI));

        return authentication == null || isPublicPath || isNonSensitivePath;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            String sid = request.getHeader("sid");
            String userId = SecurityUtil.getCurrentStudentId();

            if (userId == null) {
                throw new CustomException(AuthErrorCode.UNAUTHORIZED);
            }

            if (sid == null || !isValidSid(sid, userId)) {
                throw new CustomException(AuthErrorCode.SID_INVALID_OR_EXPIRED);
            }
            redisUtil.updateSidExpiration(userId);
        } catch (CustomException e) {
            handleException(response, e.getErrorCode());
            return;
        }

        filterChain.doFilter(request, response);
    }

    private boolean isValidSid(String sid, String userId) {
        String storedSid = redisUtil.getSidByUserId(userId);
        return storedSid != null && storedSid.equals(sid);
    }

    private void handleException(HttpServletResponse response, ErrorCode errorCode) throws IOException {
        ResponseEntity<ErrorResponse> responseEntity = ErrorResponse.responseEntity(errorCode);
        response.setStatus(responseEntity.getStatusCode().value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(response.getOutputStream(), responseEntity.getBody());
    }
}