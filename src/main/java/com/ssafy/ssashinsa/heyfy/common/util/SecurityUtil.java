package com.ssafy.ssashinsa.heyfy.common.util;

import com.ssafy.ssashinsa.heyfy.authentication.jwt.CustomUserDetails;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class SecurityUtil {

    private static CustomUserDetails getCurrentUserDetails() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication != null && authentication.getPrincipal() instanceof CustomUserDetails) {
            return (CustomUserDetails) authentication.getPrincipal();
        }
        return null;
    }

    public static String getCurrentStudentId() {
        CustomUserDetails userDetails = getCurrentUserDetails();
        return (userDetails != null) ? userDetails.getUsername() : null;
    }

    public static String getCurrentUserEmail() {
        CustomUserDetails userDetails = getCurrentUserDetails();
        return (userDetails != null) ? userDetails.getEmail() : null;
    }
}