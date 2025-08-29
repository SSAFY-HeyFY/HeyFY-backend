package com.ssafy.ssashinsa.heyfy.authentication.config;

import com.ssafy.ssashinsa.heyfy.authentication.filter.JwtAuthenticationFilter;
import com.ssafy.ssashinsa.heyfy.authentication.filter.SidValidationFilter;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.CustomAuthenticationEntryPoint;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.JwtTokenProvider;
import com.ssafy.ssashinsa.heyfy.common.util.RedisUtil;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public JwtAuthenticationFilter jwtAuthenticationFilter(JwtTokenProvider jwtTokenProvider, UserDetailsService userDetailsService, RedisUtil redisUtil) {
        return new JwtAuthenticationFilter(jwtTokenProvider, userDetailsService, redisUtil);
    }

    @Bean
    public SidValidationFilter sidValidationFilter(RedisUtil redisUtil) {
        return new SidValidationFilter(redisUtil);
    }

    @Bean
    public SecurityFilterChain filterChain(
            HttpSecurity http,
            JwtAuthenticationFilter jwtAuthenticationFilter,
            SidValidationFilter sidValidationFilter,
            CustomAuthenticationEntryPoint customAuthenticationEntryPoint
    ) throws Exception {
        return http
                .csrf(AbstractHttpConfigurer::disable)
                .sessionManagement(session ->
                        session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )
                .httpBasic(AbstractHttpConfigurer::disable)
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(ApiPaths.PUBLIC_PATHS.toArray(new String[0])).permitAll()
                        .anyRequest().authenticated()
                )
                .exceptionHandling(authenticationManager -> authenticationManager
                        .authenticationEntryPoint(customAuthenticationEntryPoint)
                )
                .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterAfter(sidValidationFilter, JwtAuthenticationFilter.class)
                .build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(
            AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }
}